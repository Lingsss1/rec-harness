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


### >>> Self_EvolveRec-BLOCK-START: Isolate reproducible data streams and use a common cutoff inverse CDF
# Category IDs follow DEFAULT_DATASETS; epochs are one-based, batches zero-based.
# Purpose IDs: 0=users, 1=cutoffs, 2=reconstruction masks, 3=diagnostic negatives.
# These streams deliberately change the original baseline's random draws.
def data_rng(category_id, epoch, batch_id, purpose_id):
    return np.random.default_rng(
        np.random.SeedSequence([SEED, category_id, epoch, batch_id, purpose_id])
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
):
    if maxlen < 2:
        raise ValueError("Terminal-prefix training requires maxlen >= 2")
    if eligible is None:
        eligible = [u for u, history in user_train.items() if len(history) >= 2]
    if not len(eligible):
        raise ValueError("No training histories contain two or more interactions")
    # Preserve the public diagnostic signature and globally seeded defaults.
    user_rng = np.random if user_rng is None else user_rng
    mask_rng = np.random if mask_rng is None else mask_rng
    negative_rng = np.random if negative_rng is None else negative_rng
    users = user_rng.choice(eligible, size=batch_size, replace=True)
    sequence = np.zeros((batch_size, maxlen), dtype=np.int64)
    positive = np.zeros_like(sequence)
    negative = np.zeros_like(sequence)
    branch = np.zeros_like(sequence, dtype=np.int8)
    histories = []
    for row, user in enumerate(users):
        history = user_train[int(user)]
        histories.append(history)
        window = np.asarray(history[-maxlen:], dtype=np.int64)
        offset = maxlen - len(window)
        sequence[row, offset:] = window
        selected = np.flatnonzero(mask_rng.random(len(window)) < 0.15)
        if not len(selected):
            selected = np.asarray([int(mask_rng.choice(len(window)))])
        forbidden = set(history)
        available = None
        for index in selected:
            position = offset + int(index)
            positive[row, position] = int(window[index])
            sequence[row, position] = itemnum + 1
            branch[row, position] = 1
            if len(forbidden) < itemnum:
                candidate = 0
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
    rng = np.random if rng is None else rng
    sequence = np.zeros((len(histories), maxlen), dtype=np.int64)
    positive = np.zeros_like(sequence)
    target_indices = []
    history_lengths = []
    context_lengths = []
    statuses = []
    selected_probabilities = []
    for row, history in enumerate(histories):
        n = len(history)
        if n < 2:
            raise ValueError("Terminal histories must contain at least two items")
        # Exactly one common uniform per example in either experimental arm.
        u = float(rng.random())
        if n == 2:
            target_index, probability = 1, 1.0
            statuses.append("singleton_endpoint_exception")
        elif u < 0.1:
            target_index, probability = n - 1, 0.1
            statuses.append("endpoint")
        else:
            m = n - 2
            k = min(10, m)
            v = min((u - 0.1) / 0.9, np.nextafter(1.0, 0.0))
            if cutoff_mode == "uniform" or k == m:
                j = min(int(v * m), m - 1)
                conditional_probability = 1.0 / m
            else:
                # Invert the two-slope mixture CDF without allocating O(n) arrays.
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
        selected_probabilities.append(probability)
        context = list(history[max(0, target_index - (maxlen - 1)) : target_index])
        values = context + [itemnum + 1]
        sequence[row, -len(values) :] = values
        positive[row, -1] = history[target_index]
        target_indices.append(target_index)
        history_lengths.append(n)
        context_lengths.append(len(context))
    terminal_batch.last_bookkeeping = {
        "target_indices": np.asarray(target_indices, dtype=np.int64),
        "history_lengths": np.asarray(history_lengths, dtype=np.int64),
        "context_lengths": np.asarray(context_lengths, dtype=np.int64),
        "statuses": statuses,
        "selected_probabilities": np.asarray(selected_probabilities, dtype=np.float64),
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


### >>> Self_EvolveRec-BLOCK-START: Normalize each objective over every valid target without quota subsampling
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
            # Projection and CE are deterministic; encoder dropout runs outside
            # this checkpoint. Avoid saving/restoring RNG state for every chunk.
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
### >>> Self_EvolveRec-BLOCK-START: Preserve the host epoch environment fallback and enforce the ceiling
NUM_EPOCHS = min(
    200,
    int(os.environ.get("BERT4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
)
if NUM_EPOCHS < 1:
    raise ValueError("BERT4REC_EPOCHS/SASREC_EPOCHS must be positive")
### <<< Self_EvolveRec-BLOCK-END
### >>> Self_EvolveRec-BLOCK-START: Reject invalid training configuration before consuming GPU budget
EVAL_EVERY = int(os.environ.get("BERT4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("BERT4REC_PATIENCE", "20"))
if EVAL_EVERY < 1 or PATIENCE < 1:
    raise ValueError("BERT4REC_EVAL_EVERY and BERT4REC_PATIENCE must be positive")
if MAXLEN < 2 or BATCH_SIZE < 2 or BATCH_SIZE % 2:
    raise ValueError("Require maxlen >= 2 and an even batch size >= 2")
if HIDDEN_UNITS < 1 or NUM_HEADS < 1 or HIDDEN_UNITS % NUM_HEADS:
    raise ValueError(
        "Hidden size must be positive and divisible by positive head count"
    )
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

    ### >>> Self_EvolveRec-BLOCK-START: Establish a reproducible uniform control and persist measured sampling allocation
    import time
    import platform

    if BATCH_SIZE < 2 or BATCH_SIZE % 2:
        raise ValueError("The mixed objective requires an even BERT4REC_BATCH >= 2")
    if MAXLEN < 2:
        raise ValueError("Terminal-prefix training requires maxlen >= 2")
    # Each invocation trains exactly one variant and one model per category.
    # Default establishes the uniform control. A separate recent10 invocation
    # changes only the conditional nonterminal cutoff distribution.
    cutoff_mode = os.environ.get("BERT4REC_CUTOFF_MODE", "uniform")
    if cutoff_mode not in ("uniform", "recent10"):
        raise ValueError("BERT4REC_CUTOFF_MODE must be uniform or recent10")
    half_batch = BATCH_SIZE // 2
    # Generator.choice otherwise converts the entire Python user list on each
    # branch of every update. Preserve user order and sampling probabilities.
    eligible = np.asarray(
        [u for u, history in user_train.items() if len(history) >= 2],
        dtype=np.int64,
    )
    if not len(eligible):
        raise ValueError("No eligible terminal training users")
    category_id = DEFAULT_DATASETS.index(dataset_name)
    updates = reconstruction_targets = terminal_examples = 0
    reconstruction_ce_sum = terminal_ce_sum = 0.0
    target_exposure = defaultdict(int)
    status_counts = defaultdict(int)
    cutoff_histogram = defaultdict(int)
    context_histogram = defaultdict(int)
    history_histogram = defaultdict(int)
    training_item_frequency = defaultdict(int)
    for history in user_train.values():
        for item in history:
            training_item_frequency[int(item)] += 1
    expected_endpoint_selections = 0.0
    selected_probability_sum = 0.0
    protocol = {
        "dataset": dataset_name,
        "variant": cutoff_mode,
        "training_seed": SEED,
        "category_id": category_id,
        "category_order": DEFAULT_DATASETS,
        "data_rng": "SeedSequence([42, category_id, one_based_epoch, zero_based_batch, purpose_id])",
        "purpose_ids": {"users": 0, "cutoffs": 1, "masks": 2, "negatives": 3},
        "changes_original_baseline_random_draws": True,
        "user_sampling": "independent uniform eligible users with replacement for each branch",
        "user_stream_order": "reconstruction users, then terminal users",
        "ineligible_users": len(user_train) - len(eligible),
        "ineligible_users_sampled": 0,
        "terminal_probabilities": {
            "n_eq_2_endpoint": 1.0,
            "n_gt_2_endpoint": 0.1,
            "n_gt_2_nonterminal": (
                "0.9/(n-2)"
                if cutoff_mode == "uniform"
                else "0.9*(0.5/(n-2)+0.5/min(10,n-2)*I[t>=n-1-min(10,n-2)])"
            ),
        },
        "objective_coefficients": {"reconstruction": 0.5, "terminal": 0.5},
        "normalization": "separate means over all valid targets in each branch",
        "output_head": {
            "denominator": "all real items 1..itemnum; PAD and MASK excluded",
            "labels": "real item ID minus one",
            "weights": "tied item_emb.weight[1:itemnum+1]",
            "bias": False,
            "reduction": "summed CE divided by branch valid-target count",
            "negative_sampling_in_loss": False,
        },
        "reconstruction": "latest maxlen events; 0.15 MASK-only with one-target fallback",
        "terminal_context": "at most maxlen-1 preceding events followed by MASK",
        "sequences_per_branch": half_batch,
        "epochs_max": NUM_EPOCHS,
        "batches_per_epoch": NUM_BATCHES,
        "eval_every": EVAL_EVERY,
        "patience": PATIENCE,
        "checkpoint_selection": "earliest maximum category validation HR@10",
        "comparison_metric": "unchanged host macro validation HR@10",
        "hyperparameters": {
            "maxlen": MAXLEN,
            "hidden_units": HIDDEN_UNITS,
            "num_blocks": NUM_BLOCKS,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT_RATE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "gradient_clip": 5.0,
        },
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
    started = time.perf_counter()
    gpu_start = gpu_end = None
    if device.type == "cuda":
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    print(
        f"  [{dataset_name}] objective=full_softmax_constant_mixture "
        f"terminal_weight=0.5 cutoff_mode={cutoff_mode} "
        f"final_target_probability_n_gt_2=0.1 recent_window=10 "
        f"reconstruction_selection=0.15 replacement=MASK_only "
        f"minimum_targets_per_window=1 eligible_history_length>=2 "
        f"sequences_per_objective_per_update={half_batch} "
        f"normalization=all_valid_targets_per_objective "
        f"projection_chunk=64 epochs={NUM_EPOCHS} "
        f"batches_per_epoch={NUM_BATCHES} eval_every={EVAL_EVERY} "
        f"patience={PATIENCE} seed={SEED} "
        f"primary_comparisons=corrected_baseline,uniform,recent10 "
        f"selection_metric=macro_validation_HR@10 checkpoint_ties=earliest",
        flush=True,
    )
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        reconstruction_sum = terminal_sum = 0.0
        epoch_reconstruction_targets = epoch_terminal_examples = 0
        final_targets = final_eligible = final_selected_eligible = 0
        history_length_sum = context_length_sum = 0
        for batch_id in range(NUM_BATCHES):
            user_rng = data_rng(category_id, epoch, batch_id, 0)
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                half_batch,
                MAXLEN,
                user_rng=user_rng,
                mask_rng=data_rng(category_id, epoch, batch_id, 2),
                negative_rng=data_rng(category_id, epoch, batch_id, 3),
                eligible=eligible,
            )
            # Keep the supplied independent branch-user allocation. Both arms
            # consume the same user stream in the same order.
            terminal_histories = [
                user_train[int(user)]
                for user in user_rng.choice(eligible, size=half_batch, replace=True)
            ]
            terminal_seq, terminal_pos = terminal_batch(
                terminal_histories,
                itemnum,
                MAXLEN,
                cutoff_mode=cutoff_mode,
                rng=data_rng(category_id, epoch, batch_id, 1),
            )
            bookkeeping = terminal_batch.last_bookkeeping
            lengths = bookkeeping["history_lengths"]
            endpoints = bookkeeping["target_indices"] == lengths - 1
            final_targets += int(endpoints.sum())
            final_eligible += int((lengths > 2).sum())
            final_selected_eligible += int((endpoints & (lengths > 2)).sum())
            history_length_sum += int(lengths.sum())
            context_length_sum += int(bookkeeping["context_lengths"].sum())
            for status in bookkeeping["statuses"]:
                status_counts[status] += 1
            for length, target_index, context_length, probability, item in zip(
                lengths,
                bookkeeping["target_indices"],
                bookkeeping["context_lengths"],
                bookkeeping["selected_probabilities"],
                terminal_pos[:, -1],
            ):
                target_exposure[int(item)] += 1
                cutoff_histogram[int(target_index)] += 1
                context_histogram[int(context_length)] += 1
                history_histogram[int(length)] += 1
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
        if gpu_end is not None:
            gpu_end.record()
            gpu_end.synchronize()
            gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000.0
        else:
            gpu_seconds = None
        print(
            json.dumps(
                {
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
                    "mean_update_objective": total_loss / NUM_BATCHES,
                    "terminal_final_fraction": final_targets / epoch_terminal_examples,
                    "terminal_final_fraction_n_gt_2": (
                        final_selected_eligible / final_eligible
                        if final_eligible
                        else None
                    ),
                    "terminal_mean_history_length": history_length_sum
                    / epoch_terminal_examples,
                    "terminal_mean_context_length": context_length_sum
                    / epoch_terminal_examples,
                    "terminal_unique_target_items": len(target_exposure),
                    "terminal_max_item_exposure": max(
                        target_exposure.values(), default=0
                    ),
                    "sequence_presentations": updates * BATCH_SIZE,
                    "encoder_token_slots": updates * BATCH_SIZE * MAXLEN,
                    "wall_seconds": time.perf_counter() - started,
                    "cuda_event_elapsed_seconds": gpu_seconds,
                }
            ),
            flush=True,
        )

        avg_loss = total_loss / NUM_BATCHES
        reconstruction_ce_sum += reconstruction_sum
        terminal_ce_sum += terminal_sum
        training_snapshot = {
            "protocol": protocol,
            "epoch": epoch,
            "optimizer_updates": updates,
            "epoch_branch_metrics": {
                "reconstruction_targets": epoch_reconstruction_targets,
                "terminal_targets": epoch_terminal_examples,
                "reconstruction_ce_sum": reconstruction_sum,
                "terminal_ce_sum": terminal_sum,
                "reconstruction_ce_per_target": reconstruction_sum
                / epoch_reconstruction_targets,
                "terminal_ce_per_target": terminal_sum / epoch_terminal_examples,
            },
            "cumulative_branch_metrics": {
                "reconstruction_examples": updates * half_batch,
                "terminal_examples": terminal_examples,
                "reconstruction_targets": reconstruction_targets,
                "terminal_targets": terminal_examples,
                "reconstruction_ce_sum": reconstruction_ce_sum,
                "terminal_ce_sum": terminal_ce_sum,
                "reconstruction_ce_per_target": reconstruction_ce_sum
                / reconstruction_targets,
                "terminal_ce_per_target": terminal_ce_sum / terminal_examples,
                "coefficient_mass_reconstruction": 0.5 * updates,
                "coefficient_mass_terminal": 0.5 * updates,
                "coefficient_mass_definition": "sum of target coefficients across updates; neither loss-value share nor gradient share",
                "empty_branches": 0,
            },
            "terminal_status_counts": dict(status_counts),
            "expected_endpoint_selections": expected_endpoint_selections,
            "realized_endpoint_selections": (
                status_counts.get("endpoint", 0)
                + status_counts.get("singleton_endpoint_exception", 0)
            ),
            "mean_probability_of_selected_target": selected_probability_sum
            / terminal_examples,
            "cutoff_index_histogram": dict(cutoff_histogram),
            "context_length_histogram": dict(context_histogram),
            "history_length_histogram": dict(history_histogram),
            "terminal_target_item_counts": dict(target_exposure),
            "training_item_frequency": dict(training_item_frequency),
            "wall_seconds": time.perf_counter() - started,
            "cuda_event_elapsed_seconds": gpu_seconds,
        }
        model.training_branch_allocation = training_snapshot
        with open(
            os.path.join(working_dir, f"{dataset_name}_training_diagnostics.json"), "w"
        ) as f:
            json.dump(training_snapshot, f)
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist allocation measured at the selected checkpoint
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
                checkpoint_diagnostics = dict(training_snapshot)
                checkpoint_diagnostics["selected_epoch"] = epoch
                checkpoint_diagnostics["selection_validation_hr10"] = hr10
                checkpoint_diagnostics["validation_evaluations"] = (
                    epoch // EVAL_EVERY + int(epoch % EVAL_EVERY != 0)
                )
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

    ### >>> Self_EvolveRec-BLOCK-START: Require a selected checkpoint and restore matching diagnostic allocation
    # Every successful run evaluates at its final executed epoch. A missing
    # selection indicates an error rather than permission to deploy final weights.
    if best_epoch < 1 or not os.path.isfile(ckpt_path):
        raise RuntimeError(f"No validation-selected checkpoint for {dataset_name}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    diagnostics_path = os.path.join(
        working_dir, f"{dataset_name}_checkpoint_diagnostics.json"
    )
    with open(diagnostics_path) as f:
        model.training_branch_allocation = json.load(f)
    model.eval()
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

### >>> Self_EvolveRec-BLOCK-START: Load measured checkpoint allocation alongside the sole category model
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
        ### >>> Self_EvolveRec-BLOCK-START: Restore diagnostic allocation without altering neural scores
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        diagnostics_path = os.path.join(working_dir, f"{ds}_checkpoint_diagnostics.json")
        if os.path.isfile(diagnostics_path):
            with open(diagnostics_path) as f:
                model.training_branch_allocation = json.load(f)
        else:
            model.training_branch_allocation = None
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the view constructor contract and resolve importlib-loaded samplers
    def __init__(self, view, device):
        import inspect

        self.model = view
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.module = view.model
        defining_module = inspect.getmodule(type(self.module))
        self.sampler = getattr(defining_module, "sample_batch", None)
        if self.sampler is None:
            encoder = getattr(type(self.module), "encode_sequence", None)
            self.sampler = getattr(encoder, "__globals__", {}).get("sample_batch")
        if self.sampler is not None and not callable(self.sampler):
            raise TypeError("The defining sample_batch must be callable")

    ### <<< Self_EvolveRec-BLOCK-END

    def _get_categories(self, item_id: int):
        ### >>> Self_EvolveRec-BLOCK-START: Mark unavailable item categories without inventing metadata
        return []
        ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Empty cosine probes are unavailable
            if n <= 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

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
        ### >>> Self_EvolveRec-BLOCK-START: Separate target coverage from valid-negative margin coverage
        was_training = self.module.training
        self.module.eval()
        observed = {
            "supervised_positions": 0,
            "mask_input": 0,
            "different_real_input": 0,
            "same_real_input": 0,
            "empty_batches": 0,
            "presented_sequences": 0,
            "valid_target_positions": 0,
            "terminal_position_targets": 0,
        }
        margins_all = []
        matched_branches = {"mask": 0, "random": 0, "unchanged": 0}
        bookkeeping_batches = 0
        unmatched_batches = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume exactly the requested diagnostic batches and reject malformed input
        from itertools import islice

        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                raise ValueError(
                    "Diagnosis requires (None, sequence, positive, negative) batches"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic tensors and consume only exactly matched bookkeeping
            seq_np = (
                seq.detach().cpu().numpy()
                if isinstance(seq, torch.Tensor)
                else np.asarray(seq)
            )
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_np.ndim != 2
                or tuple(seq_np.shape) != tuple(pos_t.shape)
                or pos_t.shape != neg_t.shape
            ):
                raise ValueError(
                    "Diagnosis requires matching B x L sequence/positive/negative tensors"
                )
            if np.any((seq_np < 0) | (seq_np > self.module.mask_token)):
                raise ValueError("Diagnostic sequence contains an invalid token")
            if torch.any((pos_t < 0) | (pos_t > self.module.itemnum)) or torch.any(
                (neg_t < 0) | (neg_t > self.module.itemnum)
            ):
                raise ValueError("Diagnostic targets must be padding or real item IDs")

            book = getattr(self.sampler, "last_bookkeeping", None)
            matched = isinstance(book, dict) and all(
                key in book and np.array_equal(book[key], value)
                for key, value in (
                    ("sequence", seq_np),
                    ("positive", pos_t.detach().cpu().numpy()),
                    ("negative", neg_t.detach().cpu().numpy()),
                )
            )
            if matched:
                branch = np.asarray(book["corruption_branch"])
                if branch.shape != seq_np.shape:
                    raise ValueError("Sampler bookkeeping shape mismatch")
                supervised_np = pos_t.detach().cpu().numpy() != 0
                if np.any(branch[supervised_np] < 1) or np.any(
                    branch[supervised_np] > 3
                ):
                    raise ValueError(
                        "Missing corruption branch at a supervised position"
                    )
                for name, code in (("mask", 1), ("random", 2), ("unchanged", 3)):
                    matched_branches[name] += int(
                        ((branch == code) & supervised_np).sum()
                    )
                bookkeeping_batches += 1
            else:
                unmatched_batches += 1

            log_feats = self.model.log2feats(seq_np)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if tuple(all_logits.shape) != tuple(all_items.shape):
                raise ValueError("ModelView.predict_score must return B x L x 2 logits")
            margin = all_logits[..., 0] - all_logits[..., 1]
            if not torch.isfinite(margin[pos_t != 0]).all():
                raise FloatingPointError("Nonfinite diagnostic margins")
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Measure terminal target coverage separately from positive-negative margins
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            itemnum = self.module.itemnum
            target_valid = (pos_t >= 1) & (pos_t <= itemnum) & (seq_t != 0)
            positions = torch.arange(seq_t.shape[1], device=self.device)
            last = (
                positions.unsqueeze(0)
                .expand_as(seq_t)
                .masked_fill(seq_t == 0, -1)
                .amax(dim=1)
            )
            terminal_targets = target_valid & (
                positions.unsqueeze(0) == last.unsqueeze(1)
            )
            observed["presented_sequences"] += int(seq_t.shape[0])
            observed["valid_target_positions"] += int(target_valid.sum().item())
            observed["terminal_position_targets"] += int(terminal_targets.sum().item())
            valid = target_valid & (neg_t >= 1) & (neg_t <= itemnum)
            observed["supervised_positions"] += int(valid.sum().item())
            observed["mask_input"] += int(
                (valid & (seq_t == self.module.mask_token)).sum().item()
            )
            observed["different_real_input"] += int(
                (valid & (seq_t >= 1) & (seq_t <= itemnum) & (seq_t != pos_t))
                .sum()
                .item()
            )
            observed["same_real_input"] += int((valid & (seq_t == pos_t)).sum().item())
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())
            else:
                observed["empty_batches"] += 1
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same real-item supervision mask in all summaries
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Empty margin probes are unavailable
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

        breakdown = {
            "eps": float(eps),
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

        ### >>> Self_EvolveRec-BLOCK-START: Report measured target coverage without inferring example branches
        breakdown["category_status"] = "unavailable: item categories are not supplied"
        breakdown["observed_inputs_at_supervised_positions"] = observed
        breakdown["terminal_position_coverage"] = (
            observed["terminal_position_targets"] / observed["presented_sequences"]
            if observed["presented_sequences"]
            else None
        )
        breakdown["corruption_branch_counts"] = (
            matched_branches if bookkeeping_batches else None
        )
        breakdown["bookkeeping_matched_batches"] = bookkeeping_batches
        breakdown["bookkeeping_unmatched_batches"] = unmatched_batches
        breakdown["corruption_branch_status"] = (
            "Branch counts cover only batches whose sequence, positive and negative "
            "arrays exactly match sampler bookkeeping. Unmatched batches are "
            "unavailable; observed same-item inputs cannot identify their branches."
        )
        self.module.train(was_training)
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Separate raw norms from centered representation spectra
    @staticmethod
    def _representation_stats(features):
        if features is None or features.ndim != 2 or features.shape[0] < 2:
            return None
        x = features.detach().float()
        if not torch.isfinite(x).all():
            return None
        centered = x - x.mean(dim=0, keepdim=True)
        sigma = torch.linalg.svdvals(centered)
        total = sigma.sum()
        rankme = None
        if float(total) > 0:
            p = sigma[sigma > 0] / total
            rankme = float(torch.exp(-(p * p.log()).sum()).item())
        return {
            "sample_count": int(x.shape[0]),
            "mean_raw_norm": float(x.norm(dim=-1).mean().item()),
            "centered_variance_per_dimension": float(centered.square().mean().item()),
            "centered_singular_values": sigma.cpu().tolist(),
            "rankme": rankme,
        }

    @torch.no_grad()
    def analyze_representation_geometry(self, sample_size=256, seed=42):
        was_training = self.module.training
        self.module.eval()
        try:
            itemnum = int(self.module.itemnum)
            rng = random.Random(seed)
            item_ids = rng.sample(range(1, itemnum + 1), min(sample_size, itemnum))
            ids = torch.as_tensor(item_ids, device=self.device, dtype=torch.long)
            item_stats = self._representation_stats(self.model.item_emb(ids))
            histories = self.user_train.get("History", self.user_train)
            eligible = sorted(histories, key=str)
            eligible = [u for u in eligible if len(histories[u]) > 0]
            users = rng.sample(eligible, min(sample_size, len(eligible)))
            rows = []
            for user in users:
                history = [int(i) for i in histories[user]]
                row = np.zeros(self.module.maxlen, dtype=np.int64)
                context = (
                    history[-(self.module.maxlen - 1) :]
                    if self.module.maxlen > 1
                    else []
                )
                values = context + [self.module.mask_token]
                row[-len(values) :] = values
                rows.append(row)
            user_stats = None
            if rows:
                representations = []
                for offset in range(0, len(rows), 32):
                    seq_np = np.stack(rows[offset : offset + 32])
                    feats = self.model.log2feats(seq_np)
                    sequence = torch.as_tensor(seq_np, device=feats.device)
                    positions = torch.arange(sequence.shape[1], device=feats.device)
                    last = (
                        positions.unsqueeze(0)
                        .expand_as(sequence)
                        .masked_fill(sequence == 0, -1)
                        .amax(dim=1)
                    )
                    representations.append(
                        feats[torch.arange(feats.shape[0], device=feats.device), last]
                    )
                user_stats = self._representation_stats(torch.cat(representations))
            return {"real_items": item_stats, "appended_mask_users": user_stats}
        finally:
            self.module.train(was_training)

    def run_full_diagnosis(self, train_loader):
        print(">>> Running masked BERT4Rec diagnosis...")
        geometry = self.analyze_representation_geometry()
        ### <<< Self_EvolveRec-BLOCK-END

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        ### >>> Self_EvolveRec-BLOCK-START: Restore module mode even when a diagnostic fails
        was_training = self.module.training
        try:
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=eps, topk_categories=20
                )
            )
        finally:
            self.module.train(was_training)
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Report directional alignment and centered geometry separately
                "embedding_directional_alignment": collapse_score,
                "representation_geometry": geometry,
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Expose measured training branch allocation when available
                "training_branch_allocation": getattr(
                    self.module, "training_branch_allocation", None
                ),
                "pos_neg_margin_mean": margin_metrics["mean"],
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define geometry without equating cosine alignment to collapse
                "embedding_directional_alignment": (
                    "Mean pairwise cosine of fixed-seed sampled real-item embeddings, "
                    "mapped to [0,1]. Measures directional alignment, not conclusive "
                    "collapse or recommendation diversity. Null means unavailable."
                ),
                "representation_geometry": (
                    "Separate fixed-seed real-item and appended-MASK user samples. "
                    "Reports raw mean L2 norm, mean squared centered coordinates, "
                    "and singular values of the centered representation matrix. "
                    "RankMe = exp(-sum(p*log(p))), p=sigma/sum(sigma); zero spectra "
                    "and fewer than two samples yield unavailable rank/statistics. "
                    "User readout uses the maximum nonpadding index."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Define stochastic view allocation and exact target-weight accounting
                "training_branch_allocation": (
                    "Measured training snapshot; null if unavailable. The checkpoint "
                    "sidecar restores counters for its selected epoch and names the "
                    "cumulative training interval. Each update draws B independent "
                    "Bernoulli(0.75) decisions over generated views, preserving B total "
                    "views and the branch-conditional user/context generators. This "
                    "replaces fixed half/half paired branch counts; it does not establish "
                    "a probability-only effect on shared-window corruption. Random "
                    "reconstruction retains 0.15 MASK-only selection with fallback; "
                    "terminal views retain the supplied sampled-prefix cutoff policy. "
                    "Each nonempty branch contributes 0.5 times its own target-mean CE; "
                    "absent branches contribute zero with no weight redistribution. "
                    "Batch records give examples, valid targets, final-position targets, "
                    "actual detached CE sums, exact target denominators, per-target "
                    "coefficients, coefficient masses, and objective contributions. "
                    "Epoch and cumulative empty counts are measured. Under independent "
                    "view draws, absence probabilities are (1-p)^B for terminal and "
                    "p^B for reconstruction; protocol records both. Coefficient mass "
                    "and example shares are neither loss-value nor gradient shares. "
                    "Global CE sums divided by global denominators differ from mean "
                    "batch objectives. Diagnostic-loader final-position coverage cannot "
                    "identify training allocation or full-history endpoint sampling. "
                    "Timing and these counters are not host validation HR@10 or evidence "
                    "of improvement over the baseline."
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Define target coverage separately from margin and branch measurements
                "pos_neg_margin_category_breakdown": (
                    "Counts low-margin and wrong-rank supervised real-item positions. "
                    "Category breakdowns are unavailable because item categories are "
                    "not supplied. Observed input counts distinguish MASK, different "
                    "real item and same real item, but cannot identify exact stochastic "
                    "corruption branches without matched sampler bookkeeping. Empty "
                    "batch counts measure probe coverage; empty margin metrics are null. "
                    "The legacy supervised_positions and observed-input counters require "
                    "both a valid positive and a valid negative. valid_target_positions "
                    "counts real positive labels at nonpadding inputs independently of "
                    "negative availability. terminal_position_targets counts those labels "
                    "at each sequence's maximum nonpadding index. Terminal-position "
                    "coverage divides that count by presented_sequences; it is null "
                    "when no sequences were measured. These diagnostic-loader counts "
                    "do not establish the training branch allocation: both reconstruction "
                    "and terminal examples can supervise their final position. "
                    "These training probes are not host validation HR@10."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
