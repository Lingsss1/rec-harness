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

### >>> Self_EvolveRec-BLOCK-START: Required data boundary and fixed training seed
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


### >>> Self_EvolveRec-BLOCK-START: Balanced uniform-prefix terminal masking with explicit normalization controls
# Each invocation trains one configuration and one model per category.
# Default treatment: exactly half terminal examples, per-example BPR.
# Matched masking control: BERT4REC_MASKING=random, reduction=per_example.
# Normalization control: BERT4REC_MASKING=random, reduction=token.
# The two random controls use identical sampling code and random draws.
# Checkpoint selection remains first strict improvement in validation HR@10.
MASKING_MODE = os.environ.get("BERT4REC_MASKING", "mixture")
LOSS_REDUCTION = os.environ.get("BERT4REC_LOSS_REDUCTION", "per_example")
RECONSTRUCTION_MASK_PROBABILITY = 0.15
if MASKING_MODE not in {"mixture", "random"}:
    raise ValueError("BERT4REC_MASKING must be mixture or random")
if LOSS_REDUCTION not in {"per_example", "token"}:
    raise ValueError("BERT4REC_LOSS_REDUCTION must be per_example or token")
if MASKING_MODE == "mixture" and LOSS_REDUCTION != "per_example":
    raise ValueError("The balanced mixture requires per-example normalization")


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    if maxlen < 2 or batch_size < 1:
        raise ValueError("maxlen must be at least two and batch_size positive")
    if MASKING_MODE == "mixture" and batch_size < 2:
        raise ValueError("The balanced mixture requires batch_size >= 2")
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    # Alternate the larger branch for odd batches; never infer branch identity
    # from target counts, since random masking can also select a single target.
    call_index = getattr(sample_batch, "call_index", 0)
    terminal_count = 0
    if MASKING_MODE == "mixture":
        terminal_count = batch_size // 2 + (batch_size % 2) * (call_index % 2)
    sample_batch.call_index = call_index + 1
    sample_batch.last_terminal_count = terminal_count
    batch_seq, batch_pos, batch_neg, provenance = [], [], [], []
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        terminal = row < terminal_count
        if terminal:
            endpoint = random.randint(1, len(history) - 1)
            start = max(0, endpoint - (maxlen - 1))
            # Target and all later events are absent from the input context.
            items = history[start:endpoint] + [history[endpoint]]
        else:
            endpoint = len(history) - 1
            start = max(0, len(history) - maxlen)
            # Random reconstruction uses the complete retained training window,
            # without the previous recency-weighted prefix selection.
            items = history[start:]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        seq[-len(items) :] = items
        offset = maxlen - len(items)
        eligible = list(range(offset, maxlen))
        if terminal:
            selected = [maxlen - 1]
        else:
            selected = [
                i for i in eligible if random.random() < RECONSTRUCTION_MASK_PROBABILITY
            ]
            if not selected:
                selected = [random.choice(eligible)]
        # Preserve rejection sampling of one uniform catalog negative per target,
        # excluding every item in this user's full supplied training history.
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError("no unobserved negative item available")
        for i in selected:
            pos[i] = seq[i]
            seq[i] = itemnum + 1
            item = random.randint(1, itemnum)
            while item in interacted:
                item = random.randint(1, itemnum)
            neg[i] = item
        branch = "terminal" if terminal else "random"
        provenance.append(
            {
                "user": int(user),
                "branch": branch,
                "cutoff": endpoint,
                "target_positions": [start + i - offset for i in selected],
                "retained_length": len(items),
                "visible_context_length": len(items) - len(selected),
                "corruption_count": len(selected),
            }
        )
        stats = sample_batch.epoch_stats.setdefault(
            branch,
            {
                "examples": 0,
                "targets": 0,
                "cutoff_sum": 0,
                "retained_length_sum": 0,
                "visible_context_length_sum": 0,
                "corruption_count": 0,
            },
        )
        stats["examples"] += 1
        stats["targets"] += len(selected)
        stats["cutoff_sum"] += endpoint
        stats["retained_length_sum"] += len(items)
        stats["visible_context_length_sum"] += len(items) - len(selected)
        stats["corruption_count"] += len(selected)
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    # Side-channel provenance preserves the three-array sampler and four-tuple
    # diagnostic-loader interfaces. Only the latest batch is retained in memory.
    sample_batch.last_provenance = provenance
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


sample_batch.epoch_stats = {}
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
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible validation without consuming training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Fixed unique validation negatives with bounded sampling
            available_count = itemnum - sum(1 <= item <= itemnum for item in interacted)
            if available_count < 99:
                raise RuntimeError(
                    f"validation user {user} has only {available_count} eligible negatives; "
                    "99 unique negatives are required"
                )
            candidates = [target]
            if available_count < 198:
                available = [
                    item for item in range(1, itemnum + 1) if item not in interacted
                ]
                candidates.extend(rng.sample(available, 99))
            else:
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


### >>> Self_EvolveRec-BLOCK-START: Implement normalization controls and explicit balanced-branch BPR
def bpr_sequence_loss(
    model: nn.Module,
    seq: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    terminal_count: int | None = None,
) -> torch.Tensor:
    # Branch membership is supplied by the training caller, never inferred
    # from target counts or mutable sampler state inside this function.
    # Diagnostic callers may omit it to obtain ordinary per-example BPR.
    if terminal_count is not None:
        if not 0 <= terminal_count < pos.shape[0]:
            raise ValueError("terminal_count must leave at least one random example")
        if terminal_count > 0 and LOSS_REDUCTION != "per_example":
            raise ValueError("Balanced branches require per-example normalization")

    pos_scores, neg_scores = model(seq, pos, neg)
    valid = pos != 0
    differences = (neg_scores - pos_scores).masked_fill(~valid, 0.0)
    losses = torch.nn.functional.softplus(differences).masked_fill(~valid, 0.0)
    counts = valid.sum(dim=1)
    if LOSS_REDUCTION == "token":
        return losses.sum() / counts.sum().clamp(min=1)

    example_losses = losses.sum(dim=1) / counts.clamp(min=1)
    if terminal_count is not None and terminal_count > 0:
        return (
            0.5 * example_losses[:terminal_count].mean()
            + 0.5 * example_losses[terminal_count:].mean()
        )
    return example_losses.mean()


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
### >>> Self_EvolveRec-BLOCK-START: Enforce the matched training budget
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1,
        int(os.environ.get("BERT4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("BERT4REC_EVAL_EVERY", "5"))
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


### >>> Self_EvolveRec-BLOCK-START: Initialize every independent category model with seed 42
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

    ### >>> Self_EvolveRec-BLOCK-START: Reset sampler allocation and declare the category experiment
    sample_batch.call_index = 0
    sample_batch.last_terminal_count = 0
    sample_batch.last_provenance = []
    sample_batch.epoch_stats = {}
    excluded_users = sum(len(history) < 2 for history in user_train.values())
    print(
        f"  [{dataset_name}] masking={MASKING_MODE} reduction={LOSS_REDUCTION}"
        f" terminal_weight={0.5 if MASKING_MODE == 'mixture' else 0.0}"
        f" excluded_short_histories={excluded_users}"
        " checkpoint_rule=first_strict_HR@10_improvement",
        flush=True,
    )
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        sample_batch.epoch_stats = {}
        model.train()
        total_loss = 0.0
        ### <<< Self_EvolveRec-BLOCK-END
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Apply explicit branch normalization in every training update
            loss = bpr_sequence_loss(
                model,
                seq_t,
                pos_t,
                neg_t,
                terminal_count=sample_batch.last_terminal_count,
            )
            ### <<< Self_EvolveRec-BLOCK-END
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        ### >>> Self_EvolveRec-BLOCK-START: Report measured masking counts and context summaries per epoch
        avg_loss = total_loss / max(NUM_BATCHES, 1)
        print(
            f"  [{dataset_name}] epoch={epoch} sampler_stats="
            + json.dumps(sample_batch.epoch_stats, sort_keys=True),
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
        ### >>> Self_EvolveRec-BLOCK-START: Require a trained checkpoint for every configured category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing BERT4Rec metadata or checkpoint for {ds}: "
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
    ### >>> Self_EvolveRec-BLOCK-START: Explicit category routing without silent cross-category predictions
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown or unloaded dataset: {dataset_name}")
        ds = dataset_name
    elif len(_models) == 1:
        ds = next(iter(_models))
    else:
        raise ValueError("dataset_name is required when multiple categories are loaded")
    model = _models[ds]
    meta = _metas[ds]
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


