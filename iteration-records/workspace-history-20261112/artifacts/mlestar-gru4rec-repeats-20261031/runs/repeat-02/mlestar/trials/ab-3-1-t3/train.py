"""GRU4Rec output dropout ablation: one checkpoint per category."""
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

DEFAULT_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = 128
NUM_LAYERS = 1
INPUT_DROPOUT = float(os.environ.get("GRU4REC_INPUT_DROPOUT", "0.2"))
OUTPUT_DROPOUT = 0.0  # Ablation: disable only output dropout.
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))))
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
    path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(path):
        with open(path) as f:
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
        interacted = set(history)
        if len(interacted - {0}) >= itemnum:
            raise RuntimeError("no unseen training negative available")
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                negative = random.randint(1, itemnum)
                while negative in interacted:
                    negative = random.randint(1, itemnum)
                neg[idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1,
                 input_dropout=0.2, output_dropout=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=0.0)
        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        # Inputs remain left-padded externally. Move real interactions to the
        # beginning for recurrence so padding cannot warm up the hidden state.
        batch_size, width = sequence.shape
        positions = torch.arange(width, device=sequence.device).unsqueeze(0)
        lengths = sequence.ne(0).sum(dim=1)
        offsets = width - lengths
        source_positions = (positions + offsets.unsqueeze(1)).clamp(max=width - 1)
        compact = sequence.gather(1, source_positions)
        compact = compact.masked_fill(positions >= lengths.unsqueeze(1), 0)
        embedded = self.input_dropout(self.item_emb(compact))
        compact_features, _ = self.gru(embedded)
        # Trailing padding is causally after every valid feature and cannot
        # affect it. Restore the original positions for sequence losses and
        # the family-specific diagnostic interface.
        feature_positions = (positions - offsets.unsqueeze(1)).clamp(min=0)
        features = compact_features.gather(
            1, feature_positions.unsqueeze(-1).expand(batch_size, width, self.hidden_units))
        features = features.masked_fill(sequence.eq(0).unsqueeze(-1), 0.0)
        # Drop outputs only; recurrent hidden states are never replaced by dropped states.
        return self.output_dropout(features)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return (features * embeddings).sum(dim=-1)

    def encode(self, seq):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        if candidates.ndim == 1:
            return torch.matmul(self.item_emb(candidates), features[0])
        return self.predict_score(features, candidates)


class ModelView(nn.Module):
    """Family-specific diagnostic interface, sharing the trained model."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def item_emb(self):
        return self.model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)

    def forward(self, seq, pos, neg):
        return self.model(seq, pos, neg)


def bpr_sequence_loss(model, seq, pos, neg):
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _history_to_tensor(history, maxlen, device):
    sequence = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    return torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = random.Random(SEED).sample(users, num_users)
    # Fixed validation negatives do not consume or perturb training RNG state.
    rng = random.Random(SEED)
    totals = {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            available = itemnum - len({i for i in excluded if 1 <= i <= itemnum})
            if available < 99:
                raise RuntimeError("validation requires 99 unique unseen negative candidates")
            candidates = [target]
            while len(candidates) < 100:
                negative = rng.randint(1, itemnum)
                if negative not in excluded:
                    excluded.add(negative)
                    candidates.append(negative)
            seq = _history_to_tensor(history, maxlen, device)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f"HR@{k}"] += 1.0
                    totals[f"NDCG@{k}"] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(len(users), 1) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")
    print(f"[{dataset_name}] users={len(user_train)} items={itemnum} input_dropout={INPUT_DROPOUT} output_dropout={OUTPUT_DROPOUT}", flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS,
                         INPUT_DROPOUT, OUTPUT_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
    best_hr10, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            batch = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(x, dtype=torch.long, device=device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss for {dataset_name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics["HR@10"]
            print(f"[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.4f} val_score: {hr10:.4f} NDCG@10={metrics['NDCG@10']:.4f}", flush=True)
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError("no evaluated checkpoint was produced")
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, input_dropout=INPUT_DROPOUT,
                output_dropout=OUTPUT_DROPOUT, best_epoch=best_epoch)
    with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[{dataset_name}] best validation HR@10={best_hr10:.4f} epoch={best_epoch}", flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""One independently trained GRU4Rec model per Amazon category."""
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


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.join(_HERE, "..", "train.py")
    spec = importlib.util.spec_from_file_location(f"gru4rec_dropout_train_{os.getpid()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dataset in DATASETS:
        if dataset in _models:
            continue
        with open(os.path.join(_HERE, f"{dataset}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = module.GRU4RecModel(
            itemnum=meta["itemnum"], hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"], input_dropout=meta["input_dropout"],
            output_dropout=meta["output_dropout"]).to(_device)
        state = torch.load(os.path.join(_HERE, f"{dataset}_model.pt"),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name is None:
        dataset_name = DATASETS[0]
    if dataset_name not in _models:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    if not candidates:
        return []
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = meta["maxlen"]
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, cand).cpu().tolist()
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
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f"All datasets done. Average val HR@10 = {sum(scores) / len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
