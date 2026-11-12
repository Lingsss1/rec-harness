"""Per-category GRU4Rec with locked input dropout; output dropout ablated."""
from __future__ import annotations

import importlib.util
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
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42

MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = 128
NUM_LAYERS = 1
INPUT_DROPOUT = 0.2
OUTPUT_DROPOUT = 0.0
LR = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get(
    "GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))))
EVAL_EVERY = 5
PATIENCE = 20


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir, dataset_name):
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]["itemnum"])
    itemnum = 0
    for split in ("train", "valid"):
        path = os.path.join(data_dir, f"{dataset_name}_{split}.txt")
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError("no unobserved training negative available")
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in interacted:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


class LockedDropout(nn.Module):
    """An independent mask per sequence, shared across its time positions."""
    def __init__(self, p):
        super().__init__()
        self.p = float(p)
        if not 0.0 <= self.p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(keep).div_(keep)
        return x * mask


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1,
                 dropout_rate=0.1, input_dropout=0.2, output_dropout=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.input_dropout = LockedDropout(input_dropout)
        self.output_dropout = LockedDropout(output_dropout)
        self.gru = nn.GRU(
            hidden_units, hidden_units, num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(
                sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        emb = self.input_dropout(self.item_emb(sequence))
        out, _ = self.gru(emb)
        return self.output_dropout(out)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        else:
            items = items.to(device=features.device, dtype=torch.long)
        item_features = self.item_emb(items)
        if features.dim() == 1:
            return torch.matmul(item_features, features)
        if items.dim() == 1:
            return torch.matmul(features, item_features.transpose(0, 1))
        if items.dim() == features.dim() - 1:
            return (features * item_features).sum(dim=-1)
        if items.dim() == features.dim():
            return (features.unsqueeze(-2) * item_features).sum(dim=-1)
        raise ValueError("unsupported feature/item shapes")

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq)[0], candidates)


def bpr_sequence_loss(model, seq, pos, neg):
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # A private RNG gives every checkpoint the same validation candidates without
    # perturbing training sampling. The host independently computes its metric.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    if not users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}
    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            available = itemnum - sum(1 for x in excluded if 1 <= x <= itemnum)
            if available < 99:
                raise RuntimeError("validation requires 99 unique eligible negatives")
            while len(candidates) < 100:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
            seq = _history_to_tensor(history, maxlen, device)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += float(rank < 10)
            hr20 += float(rank < 20)
            if rank < 10:
                ndcg10 += 1.0 / math.log2(rank + 2)
            if rank < 20:
                ndcg20 += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    n = float(len(users))
    return {"HR@10": hr10 / n, "HR@20": hr20 / n,
            "NDCG@10": ndcg10 / n, "NDCG@20": ndcg20 / n}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"[{dataset_name}] users={len(user_train)} items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or vocabulary: {dataset_name}")
    model = GRU4RecModel(
        itemnum, hidden_units=HIDDEN_UNITS, num_layers=NUM_LAYERS,
        dropout_rate=0.1, input_dropout=INPUT_DROPOUT,
        output_dropout=OUTPUT_DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
    best_hr10, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss in {dataset_name} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(
                f"[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} "
                f"loss={total_loss / NUM_BATCHES:.4f} val_score: {hr10:.4f} "
                f"HR@20={metrics['HR@20']:.4f} NDCG@10={metrics['NDCG@10']:.4f}",
                flush=True,
            )
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f"[{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError("no checkpoint selected")
    meta = {
        "itemnum": itemnum, "maxlen": MAXLEN,
        "hidden_units": HIDDEN_UNITS, "num_layers": NUM_LAYERS,
        "dropout_rate": 0.1, "input_dropout": INPUT_DROPOUT,
        "output_dropout": OUTPUT_DROPOUT,
    }
    with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[{dataset_name}] Best val HR@10={best_hr10:.4f} epoch={best_epoch}", flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""One trained locked-dropout GRU4Rec checkpoint per category."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
_models = {}
_metas = {}
_device = None


def _load_model_class():
    path = os.path.join(_HERE, "..", "train.py")
    spec = importlib.util.spec_from_file_location("locked_gru4rec_architecture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GRU4RecModel


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Model = _load_model_class()
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = Model(
            itemnum=meta["itemnum"], hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"], dropout_rate=meta["dropout_rate"],
            input_dropout=meta["input_dropout"], output_dropout=meta["output_dropout"],
        ).to(_device)
        checkpoint = torch.load(os.path.join(_HERE, f"{ds}_model.pt"),
                                map_location=_device, weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f"unknown dataset: {ds}")
    if len(candidates) == 0:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta["maxlen"])
    seq = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, "predict.py"), "w") as f:
        f.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    write_predict_script(working_dir)
    print(f"Locked-dropout GRU4Rec training on {device}; seed={SEED}", flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f"All datasets done. Average val HR@10 = {sum(scores) / len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
