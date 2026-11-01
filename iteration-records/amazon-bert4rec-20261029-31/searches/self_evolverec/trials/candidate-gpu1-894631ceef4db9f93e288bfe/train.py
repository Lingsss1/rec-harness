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


### >>> Self_EvolveRec-BLOCK-START: Restore family-specific masked sampling
# DEBUG: sample_batch was undefined. Preserve its three-array default return
# contract and expose training histories only when explicitly requested.
def sample_batch(user_train, itemnum, batch_size, maxlen, return_histories=False):
    if maxlen < 2:
        raise ValueError("Terminal-prefix training requires maxlen >= 2")
    eligible = [u for u, history in user_train.items() if len(history) >= 2]
    if not eligible:
        raise ValueError("No training histories contain two or more interactions")
    sequence = np.zeros((batch_size, maxlen), dtype=np.int64)
    positive = np.zeros_like(sequence)
    negative = np.zeros_like(sequence)
    branch = np.zeros_like(sequence, dtype=np.int8)
    histories = []
    for row in range(batch_size):
        history = user_train[random.choice(eligible)]
        histories.append(history)
        window = np.asarray(history[-maxlen:], dtype=np.int64)
        offset = maxlen - len(window)
        sequence[row, offset:] = window
        selected = np.flatnonzero(np.random.random(len(window)) < 0.15)
        if not len(selected):
            selected = np.asarray([random.randrange(len(window))])
        forbidden = set(history)
        # DEBUG: Diagnostic negatives exclude all observed training positives.
        # Zero marks unavailable negatives for saturated item vocabularies.
        available = None
        for index in selected:
            position = offset + int(index)
            positive[row, position] = int(window[index])
            sequence[row, position] = itemnum + 1
            branch[row, position] = 1
            if len(forbidden) < itemnum:
                candidate = 0
                for _ in range(64):
                    candidate = random.randint(1, itemnum)
                    if candidate not in forbidden:
                        break
                else:
                    if available is None:
                        available = [
                            i for i in range(1, itemnum + 1) if i not in forbidden
                        ]
                    candidate = random.choice(available)
                negative[row, position] = candidate
    # DEBUG: The supplied source contains no recoverable corruption policy.
    # Explicitly use 15% selection (at least one target), MASK-only replacement;
    # unchanged target identities are never supervised in bidirectional inputs.
    sample_batch.last_bookkeeping = {
        "sequence": sequence.copy(),
        "positive": positive.copy(),
        "negative": negative.copy(),
        "corruption_branch": branch.copy(),
    }
    result = (sequence, positive, negative)
    return result + (histories,) if return_histories else result


def terminal_batch(histories, itemnum, maxlen):
    sequence = np.zeros((len(histories), maxlen), dtype=np.int64)
    positive = np.zeros_like(sequence)
    for row, history in enumerate(histories):
        count = len(history) - 1
        recent_count = (count + 3) // 4
        # DEBUG: Equal-probability uniform/recent branches intentionally overlap.
        lower = 1 if random.random() < 0.5 else len(history) - recent_count
        target_index = random.randrange(lower, len(history))
        context = history[max(0, target_index - (maxlen - 1)) : target_index]
        values = context + [itemnum + 1]
        sequence[row, -len(values) :] = values
        positive[row, -1] = history[target_index]
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
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        rng = random.Random(SEED)
        val_users = rng.sample(val_users, num_users)
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            candidates = [target]
            while len(candidates) < 100 and itemnum > 0:
                neg_item = random.randint(1, itemnum)
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
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


# DEBUG: Both objectives use exact real-item full-softmax CE. Gather supervised
# states first; checkpoint projection chunks to avoid retaining catalog logits.
def masked_catalog_loss(model, seq, positive, target_quota=None):
    from torch.utils.checkpoint import checkpoint
    import torch.nn.functional as F

    selected = torch.nonzero(positive.reshape(-1) != 0, as_tuple=False).flatten()
    if selected.numel() == 0:
        raise ValueError("Masked training batch has no supervised targets")
    if target_quota is not None:
        if selected.numel() < target_quota:
            raise ValueError("Insufficient targets for the configured target quota")
        order = torch.randperm(selected.numel(), device=selected.device)
        selected = selected[order[:target_quota]]
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
        )
    return loss / selected.numel(), int(selected.numel())


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
NUM_EPOCHS = min(200, int(os.environ.get("BERT4REC_EPOCHS", "200")))
if NUM_EPOCHS < 1:
    raise ValueError("BERT4REC_EPOCHS must be positive")
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


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
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

    # DEBUG: Execute the proposed two-view constant-mixture experiment rather
    # than leaving terminal-prefix construction unreachable. This is an explicit
    # full-softmax variant; baseline equivalence or validation gains are unmeasured.
    import time

    target_quota = BATCH_SIZE
    updates = reconstruction_targets = terminal_examples = 0
    started = time.perf_counter()
    gpu_start = gpu_end = None
    if device.type == "cuda":
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    print(
        f"  [{dataset_name}] objective=full_softmax_constant_mixture "
        f"terminal_weight=0.5 endpoint_policy=equal_uniform_recent "
        f"reconstruction_selection=0.15 replacement=MASK_only "
        f"minimum_targets_per_window=1 eligible_history_length>=2 "
        f"targets_per_objective_per_update={target_quota} "
        f"projection_chunk=64 epochs={NUM_EPOCHS} "
        f"batches_per_epoch={NUM_BATCHES} eval_every={EVAL_EVERY} "
        f"patience={PATIENCE} seed={SEED}",
        flush=True,
    )
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg, histories = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, return_histories=True
            )
            terminal_seq, terminal_pos = terminal_batch(histories, itemnum, MAXLEN)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            terminal_seq_t = torch.as_tensor(
                terminal_seq, dtype=torch.long, device=device
            )
            terminal_pos_t = torch.as_tensor(
                terminal_pos, dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction_loss, count = masked_catalog_loss(
                model, seq_t, pos_t, target_quota
            )
            # DEBUG: Backpropagate the two views separately to free each graph;
            # their gradients implement the same jointly trained 0.5/0.5 loss.
            reconstruction_value = float(reconstruction_loss.detach())
            (0.5 * reconstruction_loss).backward()
            terminal_loss, terminal_count = masked_catalog_loss(
                model, terminal_seq_t, terminal_pos_t, target_quota
            )
            terminal_value = float(terminal_loss.detach())
            (0.5 * terminal_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += 0.5 * (reconstruction_value + terminal_value)
            updates += 1
            reconstruction_targets += count
            terminal_examples += terminal_count
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
                    "epoch": epoch,
                    "optimizer_updates": updates,
                    "reconstruction_targets": reconstruction_targets,
                    "terminal_examples": terminal_examples,
                    "sequence_presentations": updates * BATCH_SIZE * 2,
                    "encoder_token_slots": updates * BATCH_SIZE * MAXLEN * 2,
                    "wall_seconds": time.perf_counter() - started,
                    "cuda_event_elapsed_seconds": gpu_seconds,
                }
            ),
            flush=True,
        )

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
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
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
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    else:
        return [0.0] * len(candidates)

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


