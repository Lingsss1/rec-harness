"""Per-dataset GRU4Rec baseline for Amazon Reviews.

Trains one independent GRU4Rec checkpoint per dataset while sharing one default
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
### >>> Self_EvolveRec-BLOCK-START: Preserve four-category routing and fixed training seed
DATASETS = list(DEFAULT_DATASETS)
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


### >>> Self_EvolveRec-BLOCK-START: Uniform-endpoint final-target sampling with training-only suffix cropping
# Default calls produce reproducible, unaugmented final-target diagnostic examples.
# The training call explicitly supplies its independent augmentation stream.
def sample_batch(user_train: dict[int, list[int]], itemnum: int, batch_size: int,
                 maxlen: int, augmentation_rng=None, apply_crop: bool = False,
                 exposure=None, eligible_users=None):
    if maxlen < 1:
        raise ValueError("maxlen must be positive")
    users = eligible_users
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        # This explicitly replaces dense supervision with uniform-user,
        # uniform-training-endpoint supervision in both matched conditions.
        endpoint = random.randrange(1, len(history))
        target = history[endpoint]
        original = history[max(0, endpoint - maxlen):endpoint]
        interacted = set(history)
        neg_item = 0
        if len(interacted) < itemnum:
            # Negatives are sampled before augmentation, excluding the complete
            # training history, exactly as in the original sampler.
            for _attempt in range(32):
                neg_item = random.randint(1, itemnum)
                if neg_item not in interacted:
                    break
            else:
                pool = [i for i in range(1, itemnum + 1) if i not in interacted]
                neg_item = random.choice(pool)
        n = len(original)
        selected, k = False, n
        if augmentation_rng is not None:
            # Consume both draws even for the matched full-prefix control.
            selected = augmentation_rng.random() < 0.25
            k = augmentation_rng.randint(min(5, n), n)
        observed = original[-k:] if apply_crop and selected else original
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        seq[-len(observed):] = observed
        pos[-1] = target
        neg[-1] = neg_item
        if exposure is not None:
            exposure["rows"] += 1
            exposure["eligible_targets"] += int(neg_item != 0)
            exposure["shortened_rows"] += int(len(observed) < n)
            exposure["original_events"] += n
            exposure["retained_events"] += len(observed)
            exposure["retained_length_counts"][len(observed)] += 1
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)
### <<< Self_EvolveRec-BLOCK-END


def _history_to_tensor(history: list[int], maxlen: int, device: torch.device) -> torch.Tensor:
    seq_arr = np.zeros(maxlen, dtype=np.int64)
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    return torch.tensor([seq_arr], dtype=torch.long, device=device)


def evaluate_val(model: nn.Module, user_train: dict[int, list[int]], user_valid: dict[int, list[int]],
                 itemnum: int, device: torch.device, maxlen: int, num_users: int = 200) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Fixed local validation RNG independent of training sampling
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound unique internal validation candidates by available catalog size
            candidates = [target]
            candidate_count = 1 + min(
                99, itemnum - sum(1 <= item <= itemnum for item in interacted)
            )
            while len(candidates) < candidate_count:
                neg_item = rng.randint(1, itemnum)
            ### <<< Self_EvolveRec-BLOCK-END
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
    return {"HR@10": hr10 / n, "HR@20": hr20 / n, "NDCG@10": ndcg10 / n, "NDCG@20": ndcg20 / n}


def bpr_sequence_loss(model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    ### >>> Self_EvolveRec-BLOCK-START: Stable BPR over eligible final-target rows
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = seq.ne(0) & pos.ne(0) & neg.ne(0)
    loss = -torch.nn.functional.logsigmoid(pos_scores.float() - neg_scores.float())
    return loss.masked_fill(~mask, 0.0).sum() / mask.sum().clamp(min=1)
    ### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN        = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS  = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS    = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE  = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR            = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY  = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE    = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Fixed update budget and host epoch override
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get(
    "SASREC_EPOCHS", os.environ.get("GRU4REC_EPOCHS", "200")
))))
# Set to 0 for a separately trained matched full-prefix/final-target control.
APPLY_PREFIX_CROP = os.environ.get("GRU4REC_PREFIX_CROP", "1") == "1"
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY    = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE      = int(os.environ.get("GRU4REC_PATIENCE", "20"))


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum: int, hidden_units: int = 128, num_layers: int = 1,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        dropout = dropout_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Padding-safe causal GRU features shared by training diagnosis and prediction
    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        batch, width = seq.shape
        if batch == 0 or width == 0:
            return self.item_emb.weight.new_zeros(batch, width, self.hidden_units)
        valid = seq.ne(0)
        lengths = valid.sum(dim=1)
        positions = torch.arange(width, device=seq.device).expand_as(seq)
        order = (positions + (~valid).long() * width).argsort(dim=1)
        embeddings = self.dropout(self.item_emb(seq))
        compact = embeddings.gather(
            1, order.unsqueeze(-1).expand(-1, -1, self.hidden_units)
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.clamp(min=1).cpu(),
            batch_first=True, enforce_sorted=False,
        )
        # Use the same recurrence path for training, ModelView and prediction.
        with torch.backends.cudnn.flags(enabled=False):
            packed_out, _ = self.gru(packed)
        compact_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=width
        )
        out = torch.zeros_like(compact_out).scatter(
            1, order.unsqueeze(-1).expand_as(compact_out), compact_out
        )
        return out.masked_fill(~valid.unsqueeze(-1), 0.0)

    def predict_score(self, features: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) == 0:
            return self.item_emb.weight.new_zeros(seq.size(0), self.hidden_units)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values
        features = out[torch.arange(seq.size(0), device=seq.device), last.clamp(min=0)]
        return features.masked_fill(last.lt(0).unsqueeze(-1), 0.0)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.predict_score(self.encode(seq)[0], candidates)
    ### <<< Self_EvolveRec-BLOCK-END



### >>> Self_EvolveRec-BLOCK-START: Restore category trainer and initialize independent augmentation stream
 def_placeholder = None
### <<< Self_EvolveRec-BLOCK-END
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

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
            print(f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}", flush=True)

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({
            "itemnum": itemnum,
            "maxlen": MAXLEN,
            "hidden_units": HIDDEN_UNITS,
            "num_layers": NUM_LAYERS,
            "dropout_rate": DROPOUT_RATE,
        }, f)
    print(f"  [{dataset_name}] Best val HR@10={max(best_hr10, 0.0):.4f}  ckpt={ckpt_path}", flush=True)
    return max(best_hr10, 0.0)


def write_predict_script(working_dir: str) -> None:
    script = r'''"""Per-dataset GRU4Rec predict function."""
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
        f"gru4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


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
            num_layers=meta["num_layers"],
            dropout_rate=0.0,
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
    print(f"Per-dataset GRU4Rec training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()




