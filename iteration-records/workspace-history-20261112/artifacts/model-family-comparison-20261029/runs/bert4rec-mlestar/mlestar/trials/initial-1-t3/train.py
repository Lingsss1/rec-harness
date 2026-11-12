"""Independent per-category BERT4Rec with AdamW, warmup and linear decay."""
from __future__ import annotations

import inspect
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
DATASETS = [x.strip() for x in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",") if x.strip()]
if len(DATASETS) != 4 or set(DATASETS) != set(DEFAULT_DATASETS):
    raise ValueError("All four Amazon categories must be retained")
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MAXLEN = int(os.environ.get("BERT4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("BERT4REC_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("BERT4REC_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("BERT4REC_HEADS", "2"))
DROPOUT_RATE = float(os.environ.get("BERT4REC_DROPOUT", "0.2"))
LR = 5e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = int(os.environ.get("BERT4REC_BATCH", "128"))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get("BERT4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))))
EVAL_EVERY = 5
PATIENCE = 20


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
    return (
        _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt")),
        _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt")),
        _load_itemnum(data_dir, dataset_name),
    )


def sample_batch(user_train, itemnum, batch_size, maxlen):
    # Only masked positions are targets. No shifted targets are visible.
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        items = history[-maxlen:]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        seq[-len(items):] = items
        eligible = list(range(maxlen - len(items), maxlen))
        selected = [i for i in eligible if random.random() < 0.2]
        if not selected:
            selected = [random.choice(eligible)]
        if random.random() < 0.5 and maxlen - 1 not in selected:
            selected.append(maxlen - 1)
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
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


def _history_to_tensor(history, maxlen, device):
    seq_arr = np.zeros(maxlen, dtype=np.int64)
    items = history[-maxlen:]
    if items:
        seq_arr[-len(items):] = items
    return torch.tensor(seq_arr[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
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
            hr10 += float(rank < 10)
            hr20 += float(rank < 20)
            ndcg10 += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
            ndcg20 += 1.0 / math.log2(rank + 2) if rank < 20 else 0.0
    n = float(len(val_users))
    model.train()
    return {"HR@10": hr10 / n, "HR@20": hr20 / n, "NDCG@10": ndcg10 / n, "NDCG@20": ndcg20 / n}


class BERT4RecModel(nn.Module):
    def __init__(self, itemnum: int, hidden_units: int = 64, num_blocks: int = 2,
                 num_heads: int = 2, dropout_rate: float = 0.2, maxlen: int = 128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.mask_token = itemnum + 1
        self.item_emb = nn.Embedding(itemnum + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, nhead=num_heads,
            dim_feedforward=hidden_units * 4, dropout=dropout_rate,
            activation="gelu", batch_first=True, norm_first=True,
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
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(pos)
        x = self.dropout(x)
        padding = seq == 0
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            items = items.to(device=self.item_emb.weight.device, dtype=torch.long)
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        out = self.encode_sequence(prediction_seq)
        # Histories are left padded; the appended mask is the final nonpadding token.
        user_emb = out[0, -1]
        return torch.matmul(self.item_emb(candidates), user_emb)


def bpr_sequence_loss(model, seq, pos, neg):
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)} items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")
    model = BERT4RecModel(itemnum, HIDDEN_UNITS, NUM_BLOCKS, NUM_HEADS, DROPOUT_RATE, MAXLEN).to(device)
    decay_parameters, no_decay_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        destination = no_decay_parameters if parameter.ndim <= 1 else decay_parameters
        destination.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=LR, betas=(0.9, 0.999), eps=1e-6,
    )
    total_updates = NUM_EPOCHS * NUM_BATCHES
    warmup_updates = min(1000, max(1, total_updates // 10))
    update_index = 0
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
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
            update_index += 1
            if update_index <= warmup_updates:
                learning_rate_factor = update_index / warmup_updates
            else:
                learning_rate_factor = max(0.0, (total_updates - update_index) / max(1, total_updates - warmup_updates))
            for group in optimizer.param_groups:
                group['lr'] = LR * learning_rate_factor
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / NUM_BATCHES
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m['HR@10']
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS} loss={avg_loss:.4f}"
                f" val_score: {hr10:.4f} HR@20={val_m['HR@20']:.4f}"
                f" NDCG@10={val_m['NDCG@10']:.4f} lr={optimizer.param_groups[0]['lr']:.7f}",
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
            print(f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS} loss={avg_loss:.4f}", flush=True)
    with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump({
            "itemnum": itemnum, "maxlen": MAXLEN, "hidden_units": HIDDEN_UNITS,
            "num_blocks": NUM_BLOCKS, "num_heads": NUM_HEADS, "dropout_rate": DROPOUT_RATE,
        }, f)
    print(f"  [{dataset_name}] Best val HR@10={best_hr10:.4f} ckpt={ckpt_path}", flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr10


def write_predict_script(working_dir):
    # Embed the exact trained architecture so inference needs no import of train.py.
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    model_source = inspect.getsource(BERT4RecModel)
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
DATASETS = [x.strip() for x in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",") if x.strip()]
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in DATASETS:
        with open(os.path.join(_HERE, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = BERT4RecModel(
            itemnum=meta["itemnum"], hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"], num_heads=meta["num_heads"],
            dropout_rate=0.0, maxlen=meta["maxlen"],
        ).to(_device)
        state = torch.load(os.path.join(_HERE, f"{ds}_model.pt"), map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        ds = dataset_name
    else:
        ds = next(iter(_models))
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta["maxlen"])
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    path = os.path.join(working_dir, "predict.py")
    with open(path, "w") as f:
        f.write(header + model_source + '\n' + runtime)
    print(f"predict.py written -> {path}", flush=True)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset BERT4Rec AdamW training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    write_predict_script(working_dir)
    print(f"\nAll datasets done. Average val HR@10 = {sum(scores) / len(scores):.4f}", flush=True)


if __name__ == '__main__':
    main()
