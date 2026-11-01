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

### >>> Self_EvolveRec-BLOCK-START: Fixed benchmark data boundary and training seed
DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = list(DEFAULT_DATASETS)
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


### >>> Self_EvolveRec-BLOCK-START: Matched cutoff streams with strict eligibility and optional diagnostic-only negatives
# Fixed category IDs follow DEFAULT_DATASETS. Epochs are one-based and batches
# zero-based. These streams deliberately replace the original baseline draws.
# Purpose IDs: 0=user selection, 1=cutoffs, 2=reconstruction masks, 3=negatives.
def data_rng(category_id, epoch, batch, purpose_id):
    return np.random.default_rng(
        np.random.SeedSequence([SEED, category_id, epoch, batch, purpose_id])
    )


def sample_batch(
    user_train,
    itemnum,
    batch_size,
    maxlen,
    return_histories=False,
    *,
    user_rng=None,
    mask_rng=None,
    negative_rng=None,
    eligible=None,
    generate_negatives=True,
):
    if maxlen < 2:
        raise ValueError("Terminal-prefix training requires maxlen >= 2")
    if eligible is None:
        eligible = [u for u, history in user_train.items() if len(history) >= 2]
    if not len(eligible):
        raise ValueError("No training histories contain two or more interactions")
    # Defaults retain the public diagnostic sampler signature and seeded behavior.
    user_rng = np.random if user_rng is None else user_rng
    mask_rng = np.random if mask_rng is None else mask_rng
    negative_rng = np.random if negative_rng is None else negative_rng
    sequence = np.zeros((batch_size, maxlen), dtype=np.int64)
    positive = np.zeros_like(sequence)
    negative = np.zeros_like(sequence)
    branch = np.zeros_like(sequence, dtype=np.int8)
    histories = []
    users = user_rng.choice(eligible, size=batch_size, replace=True)
    for row, user in enumerate(users):
        history = user_train[int(user)]
        histories.append(history)
        window = np.asarray(history[-maxlen:], dtype=np.int64)
        offset = maxlen - len(window)
        sequence[row, offset:] = window
        selected = np.flatnonzero(mask_rng.random(len(window)) < 0.15)
        if not len(selected):
            selected = np.asarray([int(mask_rng.choice(len(window)))])
        # Full-catalog CE does not consume negatives. Diagnostics retain the
        # original negative-generation behavior through the public default.
        forbidden = set(history) if generate_negatives else None
        available = None
        for index in selected:
            position = offset + int(index)
            positive[row, position] = int(window[index])
            sequence[row, position] = itemnum + 1
            branch[row, position] = 1
            if generate_negatives and len(forbidden) < itemnum:
                for _ in range(64):
                    candidate = int(negative_rng.choice(itemnum)) + 1
                    if candidate not in forbidden:
                        break
                else:
                    if available is None:
                        available = [
                            i for i in range(1, itemnum + 1) if i not in forbidden
                        ]
                    candidate = int(negative_rng.choice(available))
                negative[row, position] = candidate
    sample_batch.last_bookkeeping = {
        "sequence": sequence.copy(),
        "positive": positive.copy(),
        "negative": negative.copy(),
        "corruption_branch": branch.copy(),
    }
    result = (sequence, positive, negative)
    return result + (histories,) if return_histories else result


def terminal_batch(histories, itemnum, maxlen, cutoff_mode="uniform", rng=None):
    if maxlen < 2:
        raise ValueError("Terminal-prefix training requires maxlen >= 2")
    if cutoff_mode not in ("uniform", "recent10"):
        raise ValueError("cutoff_mode must be uniform or recent10")
    if not len(histories) or any(len(history) < 2 for history in histories):
        raise ValueError("Terminal batches require nonempty histories of length >= 2")
    rng = np.random if rng is None else rng
    sequence = np.zeros((len(histories), maxlen), dtype=np.int64)
    positive = np.zeros_like(sequence)
    target_indices = np.full(len(histories), -1, dtype=np.int64)
    history_lengths = np.asarray([len(h) for h in histories], dtype=np.int64)
    context_lengths = np.zeros(len(histories), dtype=np.int64)
    selected_probabilities = np.zeros(len(histories), dtype=np.float64)
    statuses = []
    for row, history in enumerate(histories):
        n = len(history)
        # Exactly one common uniform is consumed for every presented history.
        u = float(rng.random())
        if not 0.0 <= u < 1.0:
            raise ValueError("Cutoff RNG must return a finite value in [0, 1)")
        if n == 2:
            target_index, probability = 1, 1.0
            statuses.append("singleton_endpoint_exception")
        elif u < 0.1:
            target_index, probability = n - 1, 0.1
            statuses.append("endpoint")
        else:
            m = n - 2
            v = (u - 0.1) / 0.9
            k = min(10, m)
            if cutoff_mode == "uniform" or k == m:
                # When every eligible position is recent, both policies coincide.
                j = min(int(v * m), m - 1)
                conditional_probability = 1.0 / m
            else:
                # Analytic inverse of the same two-slope CDF. This avoids
                # allocating and scanning arrays proportional to history length.
                older_count = m - k
                older_probability = 0.5 / m
                recent_probability = older_probability + 0.5 / k
                older_mass = older_count * older_probability
                if v < older_mass:
                    j = min(int(v / older_probability), older_count - 1)
                    conditional_probability = older_probability
                else:
                    recent_index = min(
                        int((v - older_mass) / recent_probability), k - 1
                    )
                    j = older_count + recent_index
                    conditional_probability = recent_probability
            target_index = j + 1
            probability = 0.9 * conditional_probability
            statuses.append("nonterminal")
        context = list(history[max(0, target_index - (maxlen - 1)) : target_index])
        values = context + [itemnum + 1]
        sequence[row, -len(values) :] = values
        positive[row, -1] = history[target_index]
        target_indices[row] = target_index
        context_lengths[row] = len(context)
        selected_probabilities[row] = probability
    terminal_batch.last_bookkeeping = {
        "target_indices": target_indices,
        "history_lengths": history_lengths,
        "context_lengths": context_lengths,
        "statuses": statuses,
        "selected_probabilities": selected_probabilities,
    }
    return sequence, positive


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
    ### >>> Self_EvolveRec-BLOCK-START: Isolate deterministic validation sampling from training randomness
    rng = random.Random(SEED)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    ### >>> Self_EvolveRec-BLOCK-START: Preserve caller mode and avoid validation autograd bookkeeping
    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    was_training = model.training
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END
    with torch.inference_mode():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            ### >>> Self_EvolveRec-BLOCK-START: Require valid unique validation candidates and bound rejection sampling
            interacted = set(history) | {target, 0}
            if not 1 <= target <= itemnum:
                raise ValueError(f"Validation target outside item vocabulary: {target}")
            excluded_count = sum(1 <= item <= itemnum for item in interacted)
            if itemnum - excluded_count < 99:
                raise ValueError(
                    f"User {user} has fewer than 99 eligible validation negatives"
                )
            candidates = [target]
            attempts = 0
            while len(candidates) < 100 and attempts < 10000:
                neg_item = rng.randint(1, itemnum)
                attempts += 1
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
            if len(candidates) < 100:
                remaining = [
                    item for item in range(1, itemnum + 1) if item not in interacted
                ]
                candidates.extend(rng.sample(remaining, 100 - len(candidates)))
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
    ### >>> Self_EvolveRec-BLOCK-START: Restore the model mode supplied by the validation caller
    model.train(was_training)
    ### <<< Self_EvolveRec-BLOCK-END
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


### >>> Self_EvolveRec-BLOCK-START: Preserve target-normalized catalog losses with cheaper deterministic projection checkpointing
def masked_catalog_loss(model, seq, positive):
    from torch.utils.checkpoint import checkpoint
    import torch.nn.functional as F

    selected = torch.nonzero(positive.reshape(-1) != 0, as_tuple=False).flatten()
    if selected.numel() == 0:
        raise ValueError("Masked training batch has no supervised targets")
    features = model.encode_sequence(seq).reshape(-1, model.hidden_units)[selected]
    labels = positive.reshape(-1)[selected] - 1
    weights = model.item_emb.weight[1 : model.itemnum + 1]

    def chunk_loss(states, targets, catalog):
        return F.cross_entropy(states @ catalog.t(), targets, reduction="sum")

    loss = features.new_zeros(())
    for start in range(0, selected.numel(), 64):
        loss = loss + checkpoint(
            chunk_loss,
            features[start : start + 64],
            labels[start : start + 64],
            weights,
            use_reentrant=False,
            # Only deterministic projection and CE are recomputed here.
            # Encoder dropout remains outside this checkpointed function.
            preserve_rng_state=False,
        )
    return loss / selected.numel(), int(selected.numel())


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
# DEBUG: Keep the prescribed update ceiling and batches per epoch fixed.
NUM_BATCHES = 200
### >>> Self_EvolveRec-BLOCK-START: Honor both host and family epoch limits and reject invalid execution settings
NUM_EPOCHS = min(
    200,
    int(os.environ.get("SASREC_EPOCHS", "200")),
    int(os.environ.get("BERT4REC_EPOCHS", "200")),
)
if NUM_EPOCHS < 1:
    raise ValueError("SASREC_EPOCHS and BERT4REC_EPOCHS must be positive")
EVAL_EVERY = int(os.environ.get("BERT4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("BERT4REC_PATIENCE", "20"))
if EVAL_EVERY < 1 or PATIENCE < 1:
    raise ValueError("Evaluation interval and patience must be positive")
if MAXLEN < 2 or BATCH_SIZE < 2 or BATCH_SIZE % 2:
    raise ValueError("Require maxlen >= 2 and an even batch size >= 2")
if HIDDEN_UNITS < 1 or NUM_HEADS < 1 or HIDDEN_UNITS % NUM_HEADS:
    raise ValueError("Hidden size must be positive and divisible by head count")
if NUM_BLOCKS < 1 or not 0.0 <= DROPOUT_RATE < 1.0:
    raise ValueError("Require positive block count and dropout in [0, 1)")
if not math.isfinite(LR) or LR <= 0:
    raise ValueError("Learning rate must be finite and positive")
if not math.isfinite(WEIGHT_DECAY) or WEIGHT_DECAY < 0:
    raise ValueError("Weight decay must be finite and nonnegative")
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
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

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        out = self.encode_sequence(prediction_seq)
        user_emb = out[0, -1]
        cand_emb = self.item_emb(candidates)
        return torch.matmul(cand_emb, user_emb)


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Initialize each independent category model with training seed 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    ### <<< Self_EvolveRec-BLOCK-END
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
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

    ### >>> Self_EvolveRec-BLOCK-START: Establish the uniform control with matched streams and separately normalized objectives
    import time
    import platform

    if BATCH_SIZE < 2 or BATCH_SIZE % 2:
        raise ValueError("The mixed objective requires an even BERT4REC_BATCH >= 2")
    if MAXLEN < 2:
        raise ValueError("Terminal-prefix training requires maxlen >= 2")
    # Establish the uniform control first. A subsequent separate trial selects
    # BERT4REC_CUTOFF_MODE=recent10, changing only nonterminal cutoff probabilities.
    # Historical aggregate scores do not establish a matched comparison.
    # Each invocation trains one model per category without loading other trials.
    cutoff_mode = os.environ.get("BERT4REC_CUTOFF_MODE", "uniform")
    if cutoff_mode not in ("uniform", "recent10"):
        raise ValueError("BERT4REC_CUTOFF_MODE must be uniform or recent10")
    category_id = DEFAULT_DATASETS.index(dataset_name)
    half_batch = BATCH_SIZE // 2
    eligible = np.asarray(
        [u for u, history in user_train.items() if len(history) >= 2],
        dtype=np.int64,
    )
    if not len(eligible):
        raise ValueError("No eligible terminal training users")
    updates = reconstruction_targets = terminal_examples = 0
    status_counts = defaultdict(int)
    target_exposure = defaultdict(int)
    cutoff_histogram = defaultdict(int)
    context_histogram = defaultdict(int)
    history_histogram = defaultdict(int)
    training_item_frequency = defaultdict(int)
    for history in user_train.values():
        for item in history:
            training_item_frequency[int(item)] += 1
    expected_endpoint_selections = 0.0
    selected_probability_sum = 0.0
    reconstruction_ce_sum = terminal_ce_sum = 0.0
    started = time.perf_counter()
    gpu_start = gpu_end = None
    if device.type == "cuda":
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    protocol = {
        "dataset": dataset_name,
        "variant": cutoff_mode,
        "seed": SEED,
        "category_id": category_id,
        "category_order": DEFAULT_DATASETS,
        "data_rng": "SeedSequence([42, category_id, one_based_epoch, zero_based_batch, purpose_id])",
        "purpose_ids": {"users": 0, "cutoffs": 1, "masks": 2, "negatives": 3},
        "changes_original_baseline_random_draws": True,
        "users_shared_between_objectives": True,
        "user_sampling": "uniform eligible users with replacement",
        "ineligible_training_users": len(user_train) - len(eligible),
        "terminal_expected_probabilities": {
            "n_lt_2": "skip",
            "n_eq_2": "endpoint=1",
            "n_gt_2_endpoint": 0.1,
            "n_gt_2_nonterminal": (
                "0.9/(n-2)"
                if cutoff_mode == "uniform"
                else "0.9*(0.5/(n-2)+0.5/min(10,n-2)*I[index>=n-1-min(10,n-2)])"
            ),
        },
        "objective_coefficients": {"reconstruction": 0.5, "terminal": 0.5},
        "normalization": "separate means over all valid targets",
        "reconstruction_policy": "0.15 MASK-only, at least one target per window",
        "sequences_per_branch_per_update": half_batch,
        "epochs_max": NUM_EPOCHS,
        "batches_per_epoch": NUM_BATCHES,
        "eval_every": EVAL_EVERY,
        "patience": PATIENCE,
        "checkpoint_selection": "earliest maximum category validation HR@10",
        "comparison_metric": "host macro validation HR@10",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "data_loader_workers": 0,
    }
    print(json.dumps(protocol), flush=True)
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        reconstruction_sum = terminal_sum = 0.0
        epoch_reconstruction_targets = epoch_terminal_examples = 0
        for batch_id in range(NUM_BATCHES):
            seq, pos, neg, terminal_histories = sample_batch(
                user_train,
                itemnum,
                half_batch,
                MAXLEN,
                return_histories=True,
                user_rng=data_rng(category_id, epoch, batch_id, 0),
                mask_rng=data_rng(category_id, epoch, batch_id, 2),
                negative_rng=data_rng(category_id, epoch, batch_id, 3),
                eligible=eligible,
                generate_negatives=False,
            )
            terminal_seq, terminal_pos = terminal_batch(
                terminal_histories,
                itemnum,
                MAXLEN,
                cutoff_mode=cutoff_mode,
                rng=data_rng(category_id, epoch, batch_id, 1),
            )
            bookkeeping = terminal_batch.last_bookkeeping
            for status in bookkeeping["statuses"]:
                status_counts[status] += 1
            for length, target_index, context_length, probability, item in zip(
                bookkeeping["history_lengths"],
                bookkeeping["target_indices"],
                bookkeeping["context_lengths"],
                bookkeeping["selected_probabilities"],
                terminal_pos[:, -1],
            ):
                if target_index < 0:
                    continue
                length = int(length)
                target_exposure[int(item)] += 1
                cutoff_histogram[int(target_index)] += 1
                context_histogram[int(context_length)] += 1
                history_histogram[length] += 1
                selected_probability_sum += float(probability)
                expected_endpoint_selections += 1.0 if length == 2 else 0.1
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            terminal_seq_t = torch.as_tensor(
                terminal_seq, dtype=torch.long, device=device
            )
            terminal_pos_t = torch.as_tensor(
                terminal_pos, dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction_loss, count = masked_catalog_loss(model, seq_t, pos_t)
            reconstruction_value = float(reconstruction_loss.detach())
            (0.5 * reconstruction_loss).backward()
            terminal_loss, terminal_count = masked_catalog_loss(
                model, terminal_seq_t, terminal_pos_t
            )
            terminal_value = float(terminal_loss.detach())
            (0.5 * terminal_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += 0.5 * (reconstruction_value + terminal_value)
            reconstruction_sum += reconstruction_value * count
            terminal_sum += terminal_value * terminal_count
            updates += 1
            reconstruction_targets += count
            terminal_examples += terminal_count
            epoch_reconstruction_targets += count
            epoch_terminal_examples += terminal_count
        reconstruction_ce_sum += reconstruction_sum
        terminal_ce_sum += terminal_sum
        gpu_seconds = None
        if gpu_end is not None:
            gpu_end.record()
            gpu_end.synchronize()
            gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000.0
        singleton = status_counts.get("singleton_endpoint_exception", 0)
        endpoint = status_counts.get("endpoint", 0)
        nonterminal = status_counts.get("nonterminal", 0)
        avg_loss = total_loss / NUM_BATCHES
        epoch_metrics = {
            "dataset": dataset_name,
            "variant": cutoff_mode,
            "epoch": epoch,
            "optimizer_updates": updates,
            "reconstruction_targets": reconstruction_targets,
            "terminal_examples": terminal_examples,
            "epoch_reconstruction_targets": epoch_reconstruction_targets,
            "epoch_terminal_examples": epoch_terminal_examples,
            "reconstruction_ce_per_target": reconstruction_sum
            / epoch_reconstruction_targets,
            "terminal_ce_per_example": terminal_sum / epoch_terminal_examples,
            "cumulative_reconstruction_ce_per_target": reconstruction_ce_sum
            / reconstruction_targets,
            "cumulative_terminal_ce_per_target": terminal_ce_sum / terminal_examples,
            "mean_update_objective": avg_loss,
            "coefficient_mass_reconstruction": 0.5 * updates,
            "coefficient_mass_terminal": 0.5 * updates,
            "coefficient_mass_definition": "sum of per-target loss coefficients across updates; not loss or gradient share",
            "terminal_status_counts": dict(status_counts),
            "terminal_expected_endpoint_selections": expected_endpoint_selections,
            "terminal_realized_endpoint_selections": endpoint + singleton,
            "terminal_final_fraction_n_gt_2": (
                endpoint / (endpoint + nonterminal) if endpoint + nonterminal else None
            ),
            "terminal_mean_selected_target_probability": selected_probability_sum
            / terminal_examples,
            "terminal_mean_context_length": sum(
                k * v for k, v in context_histogram.items()
            )
            / terminal_examples,
            "terminal_unique_target_items": len(target_exposure),
            "sequence_presentations": updates * BATCH_SIZE,
            "encoder_token_slots": updates * BATCH_SIZE * MAXLEN,
            "wall_seconds": time.perf_counter() - started,
            "cuda_event_elapsed_seconds": gpu_seconds,
        }
        print(json.dumps(epoch_metrics), flush=True)
        training_snapshot = {
            "protocol": protocol,
            "metrics": epoch_metrics,
            "cutoff_index_histogram": dict(cutoff_histogram),
            "context_length_histogram": dict(context_histogram),
            "history_length_histogram": dict(history_histogram),
            "terminal_target_item_counts": dict(target_exposure),
            "training_item_frequency": dict(training_item_frequency),
        }
        with open(
            os.path.join(working_dir, f"{dataset_name}_training_diagnostics.json"), "w"
        ) as f:
            json.dump(training_snapshot, f)
        ### <<< Self_EvolveRec-BLOCK-END
        ### >>> Self_EvolveRec-BLOCK-START: Persist measured control and treatment allocation with explicit loss semantics
        protocol["output_head"] = {
            "denominator": "real items 1..itemnum; PAD and MASK excluded",
            "labels": "item_id minus one",
            "weights": "tied item_emb.weight[1:itemnum+1]",
            "bias": False,
            "negative_sampling_in_loss": False,
            "reduction": "separate branch-wide valid-target means",
        }
        protocol["hyperparameters"] = {
            "maxlen": MAXLEN,
            "hidden_units": HIDDEN_UNITS,
            "num_blocks": NUM_BLOCKS,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": 5.0,
        }
        protocol["eligible_training_users"] = int(len(eligible))
        protocol["eligible_users_where_policies_can_differ"] = sum(
            len(user_train[int(user)]) > 12 for user in eligible
        )
        protocol["terminal_expected_probabilities"][
            "n_lt_2"
        ] = "ineligible; raises if presented"
        protocol["diagnostic_negatives"] = {
            "used_in_training_loss": False,
            "generated_in_training_loop": False,
            "public_sampler_default": True,
            "sampling_rule": "uniform rejection of training-history items",
        }
        policy_examples = sum(
            count for length, count in history_histogram.items() if length > 12
        )
        training_snapshot["terminal_examples_where_policies_can_differ"] = (
            policy_examples
        )
        training_snapshot["terminal_policy_distinguishable_fraction"] = (
            policy_examples / terminal_examples
        )
        training_snapshot["expected_cutoff_distribution"] = {
            "definition": (
                "Complete per-example distribution indexed by history length: "
                "endpoint mass plus nonterminal base mass and recent-window bonus. "
                "Multiply each mass by presented_examples to obtain expected counts."
            ),
            "by_history_length": {
                str(length): {
                    "presented_examples": count,
                    "endpoint_index": length - 1,
                    "endpoint_probability": 1.0 if length == 2 else 0.1,
                    "nonterminal_first_index": 1 if length > 2 else None,
                    "nonterminal_last_index": length - 2 if length > 2 else None,
                    "nonterminal_probability_per_index": (
                        (0.9 if cutoff_mode == "uniform" else 0.45) / (length - 2)
                        if length > 2
                        else 0.0
                    ),
                    "recent_first_index": (
                        length - 1 - min(10, length - 2) if length > 2 else None
                    ),
                    "recent_last_index": length - 2 if length > 2 else None,
                    "recent_additional_probability_per_index": (
                        0.45 / min(10, length - 2)
                        if length > 2 and cutoff_mode == "recent10"
                        else 0.0
                    ),
                }
                for length, count in history_histogram.items()
            },
        }
        training_snapshot["epoch_branch_metrics"] = {
            "reconstruction_examples": NUM_BATCHES * half_batch,
            "terminal_examples": epoch_terminal_examples,
            "reconstruction_targets": epoch_reconstruction_targets,
            "terminal_targets": epoch_terminal_examples,
            "reconstruction_ce_sum": reconstruction_sum,
            "terminal_ce_sum": terminal_sum,
            "coefficient_mass_reconstruction": 0.5 * NUM_BATCHES,
            "coefficient_mass_terminal": 0.5 * NUM_BATCHES,
            "empty_branches": 0,
        }
        model.training_branch_allocation = training_snapshot
        with open(
            os.path.join(working_dir, f"{dataset_name}_training_diagnostics.json"), "w"
        ) as f:
            json.dump(training_snapshot, f)
        with open(
            os.path.join(working_dir, f"{dataset_name}_epoch_metrics.jsonl"),
            "w" if epoch == 1 else "a",
        ) as f:
            json.dump(
                {
                    "metrics": epoch_metrics,
                    "branch_metrics": training_snapshot["epoch_branch_metrics"],
                },
                f,
            )
            f.write("\n")
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist measured training allocation for the selected single checkpoint
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
                checkpoint_diagnostics = dict(training_snapshot)
                checkpoint_diagnostics["selected_epoch"] = epoch
                checkpoint_diagnostics["selection_validation_hr10"] = hr10
                with open(
                    os.path.join(
                        working_dir, f"{dataset_name}_checkpoint_diagnostics.json"
                    ),
                    "w",
                ) as f:
                    json.dump(checkpoint_diagnostics, f)
                ### <<< Self_EvolveRec-BLOCK-END
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    ### >>> Self_EvolveRec-BLOCK-START: Restore the single validation-selected model and its measured diagnostic snapshot
    if best_epoch < 1 or not os.path.isfile(ckpt_path):
        raise RuntimeError(f"No validation-selected checkpoint for {dataset_name}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    with open(
        os.path.join(working_dir, f"{dataset_name}_checkpoint_diagnostics.json")
    ) as f:
        model.training_branch_allocation = json.load(f)
    model.eval()
    with open(
        os.path.join(working_dir, f"{dataset_name}_search_summary.json"), "w"
    ) as f:
        json.dump(
            {
                "variant": cutoff_mode,
                "selected_epoch": best_epoch,
                "executed_epochs": epoch,
                "optimizer_updates": updates,
                "validation_evaluations": epoch // EVAL_EVERY
                + int(epoch == NUM_EPOCHS and epoch % EVAL_EVERY != 0),
                "best_category_validation_hr10": best_hr10,
                "checkpoint_selection": protocol["checkpoint_selection"],
                "wall_seconds": time.perf_counter() - started,
            },
            f,
        )
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
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

### >>> Self_EvolveRec-BLOCK-START: Load checkpoint diagnostic sidecars in the embedded predictor
import importlib.util
import json
import os
import pickle
### <<< Self_EvolveRec-BLOCK-END

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Preserve all four dataset routes in the generated predictor
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Require complete category loading before treating the predictor as initialized
    if all(ds in _models for ds in DATASETS):
        return
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        ### >>> Self_EvolveRec-BLOCK-START: Avoid reloading completed categories after an explicit loading failure
        if ds in _models:
            continue
        ### <<< Self_EvolveRec-BLOCK-END
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Surface missing category artifacts instead of silently dropping routes
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing BERT4Rec artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Restore measured allocation for the sole category checkpoint
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        diagnostics_path = os.path.join(working_dir, f"{ds}_checkpoint_diagnostics.json")
        with open(diagnostics_path) as f:
            model.training_branch_allocation = json.load(f)
        model.eval()
        ### <<< Self_EvolveRec-BLOCK-END
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Reject unknown dataset routes and preserve the legacy default route
    ds = DEFAULT_DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError(f"Unknown or unavailable BERT4Rec dataset: {ds}")
    model = _models[ds]
    meta = _metas[ds]
    if not candidates:
        return []
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


