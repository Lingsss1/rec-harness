from __future__ import annotations

import inspect
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "working"
SEED = 42
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "64"))
HIDDEN = int(os.environ.get("SASREC_HIDDEN", "96"))
DROPOUT = float(os.environ.get("SASREC_DROPOUT", "0.25"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
LR = float(os.environ.get("SASREC_LR", "0.001"))
NEGATIVES = 32


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                result[int(p[0])].append(int(p[1]))
    return dict(result)


def load_data(name):
    train = read_split(Path(DATA_DIR) / (name + "_train.txt"))
    valid = read_split(Path(DATA_DIR) / (name + "_valid.txt"))
    stats_file = Path(DATA_DIR) / "dataset_stats.json"
    itemnum = max(max(items, default=0) for items in train.values())
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        if name in stats:
            itemnum = int(stats[name]["itemnum"])
    if itemnum < 2:
        raise ValueError("Insufficient item vocabulary: " + name)
    return train, valid, itemnum


class RecurrentMemory(nn.Module):
    def __init__(self, itemnum, hidden=96, maxlen=64, dropout=0.25):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.embedding = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.input_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.query_projection = nn.Linear(hidden, hidden, bias=False)
        self.key_projection = nn.Linear(hidden, hidden, bias=False)
        self.value_projection = nn.Linear(hidden, hidden, bias=False)
        self.memory_projection = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(3 * hidden, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.03)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        for name, p in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def encode(self, seq):
        mask = seq.ne(0).unsqueeze(-1)
        raw = self.embedding(seq)
        x = self.input_norm(raw) * mask
        recurrent, _ = self.gru(self.dropout(x))
        # Retrieve relevant prefix items using jointly trained causal attention.
        queries = self.query_projection(recurrent)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        causal = torch.ones(
            seq.shape[1], seq.shape[1], dtype=torch.bool, device=seq.device
        ).tril()
        attention_mask = causal.unsqueeze(0) & mask.transpose(1, 2) & mask
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(queries.shape[-1])
        attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))
        # Padded queries have no allowed keys; use finite logits before softmax
        # and then zero their weights so neither outputs nor gradients see NaNs.
        attention_scores = attention_scores.masked_fill(~mask, 0.0)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(~attention_mask, 0.0)
        memory = torch.matmul(attention_weights, values) * mask
        memory = torch.tanh(self.memory_projection(memory))
        gate = torch.sigmoid(self.gate(torch.cat([recurrent, x, memory], dim=-1)))
        h = self.output_norm(x + recurrent + gate * memory)
        return self.dropout(h)

    def forward(self, seq, positives, negatives):
        h = self.encode(seq)
        pos = (h * self.embedding(positives)).sum(-1)
        pos = pos + self.item_bias(positives).squeeze(-1)
        # The same negative set is reused across positions within a history.
        neg = torch.einsum("blh,bkh->blk", h, self.embedding(negatives))
        neg = neg + self.item_bias(negatives).squeeze(-1).unsqueeze(1)
        return pos, neg

    def predict(self, seq, candidates):
        h = self.encode(seq)
        lengths = seq.ne(0).sum(1).clamp_min(1)
        final = h[torch.arange(seq.shape[0], device=seq.device), lengths - 1]
        scores = torch.einsum("bh,bkh->bk", final, self.embedding(candidates))
        return scores + self.item_bias(candidates).squeeze(-1)


class HistoryDataset(Dataset):
    def __init__(self, train, maxlen):
        self.histories = [
            np.asarray(v, dtype=np.int64) for v in train.values() if len(v) >= 2
        ]
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No training histories with next-item transitions")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        hist = self.histories[index]
        # Most examples supervise recent transitions; random prefixes also
        # expose older transitions in histories longer than the context window.
        end = len(hist)
        if len(hist) > 3 and np.random.random() < 0.25:
            end = int(np.random.randint(2, len(hist) + 1))
        window = hist[max(0, end - self.maxlen - 1) : end]
        length = len(window) - 1
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        seq[:length] = window[:-1]
        pos[:length] = window[1:]
        return seq, pos


def worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def fixed_validation(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        target = valid[user][0]
        seen = set(train[user]) | set(valid[user])
        if itemnum - len(seen) < 99:
            raise ValueError("Cannot draw 99 unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in seen:
                negatives.add(item)
        seq = np.zeros(MAXLEN, dtype=np.int64)
        hist = train[user][-MAXLEN:]
        seq[: len(hist)] = hist
        sequences.append(seq)
        candidates.append([target] + list(negatives))
    if not sequences:
        raise ValueError("No validation examples")
    return np.stack(sequences), np.asarray(candidates, dtype=np.int64)


@torch.inference_mode()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    hits, ndcg, count = 0, 0.0, 0
    for start in range(0, len(sequences), 128):
        seq = torch.as_tensor(sequences[start : start + 128], device=device)
        cand = torch.as_tensor(candidates[start : start + 128], device=device)
        scores = model.predict(seq, cand).float().cpu().numpy()
        # Stable sorting preserves the evaluator's target-first tie convention.
        ordering = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.argmax(ordering == 0, axis=1)
        hit = ranks < 10
        hits += int(hit.sum())
        ndcg += float(np.where(hit, 1.0 / np.log2(ranks + 2), 0.0).sum())
        count += len(ranks)
    return hits / count, ndcg / count


def train_category(name, device, deadline):
    seed_everything(SEED)
    train, valid, itemnum = load_data(name)
    validation = fixed_validation(train, valid, itemnum)
    data = HistoryDataset(train, MAXLEN)
    generator = torch.Generator().manual_seed(SEED)
    sampler = RandomSampler(
        data, replacement=True, num_samples=BATCH * STEPS, generator=generator
    )
    loader = DataLoader(
        data,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init,
        generator=generator,
    )
    config = dict(itemnum=itemnum, hidden=HIDDEN, maxlen=MAXLEN, dropout=DROPOUT)
    model = RecurrentMemory(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=1e-5
    )
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    checkpoint = WORK / (name + "_model.pt")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Gentle decay retains useful learning during early stopping.
        lr = LR * (0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / EPOCHS)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = 0.0
        batches = 0
        for seq_cpu, pos_cpu in loader:
            seq = seq_cpu.to(device, non_blocking=True)
            pos = pos_cpu.to(device, non_blocking=True)
            neg = torch.randint(
                1, itemnum + 1, (seq.shape[0], NEGATIVES), device=device
            )
            pos_scores, neg_scores = model(seq, pos, neg)
            valid_positions = pos.ne(0)
            # Exclude known context items and accidental positive matches.
            known = (neg.unsqueeze(1) == seq.unsqueeze(-1)).any(dim=1)
            allowed = (~known.unsqueeze(1)) & neg.unsqueeze(1).ne(pos.unsqueeze(-1))
            allowed = allowed & valid_positions.unsqueeze(-1)
            masked_neg_scores = neg_scores.masked_fill(~allowed, float("-inf"))
            candidate_logits = torch.cat(
                [pos_scores.unsqueeze(-1), masked_neg_scores], dim=-1
            )
            per_position = torch.logsumexp(candidate_logits, dim=-1) - pos_scores
            # Weight supervised transitions by distance from each row's latest target.
            position_indices = torch.arange(pos.shape[1], device=pos.device).unsqueeze(0)
            last_valid_index = position_indices.masked_fill(
                ~valid_positions, -1
            ).max(dim=1, keepdim=True).values
            distances = (last_valid_index - position_indices).clamp_min(0)
            weights = (0.9 ** distances.to(pos_scores.dtype)) * valid_positions
            weight_sum = weights.sum().clamp_min(1e-8)
            ranking_loss = (per_position * weights).sum() / weight_sum
            positive_loss = (F.softplus(-pos_scores) * weights).sum() / weight_sum
            # Normalize negatives within each position before temporal reduction.
            per_position_negative = (
                F.softplus(neg_scores) * allowed
            ).sum(-1) / allowed.sum(-1).clamp_min(1)
            negative_loss = (per_position_negative * weights).sum() / weight_sum
            loss = ranking_loss + 0.2 * (positive_loss + negative_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        hr, ndcg = evaluate(model, validation, device)
        if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
            torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)
        print(
            f"{name} epoch={epoch} loss={total_loss / max(batches, 1):.5f} "
            f"HR@10={hr:.5f} NDCG@10={ndcg:.5f}",
            flush=True,
        )
        if epoch - best_epoch >= 30 or time.monotonic() >= deadline:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    final_hr, _ = evaluate(model, validation, device)
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_hr


def write_predictor():
    # Export the exact architecture, avoiding a dependency on train.py imports.
    model_source = (
        "import math\nimport torch\nfrom torch import nn\n"
        "from torch.nn import functional as F\n\n"
        + inspect.getsource(RecurrentMemory)
        + "\n"
    )
    (WORK / "recurrent_model.py").write_text(model_source)
    predictor = r"""from __future__ import annotations
import importlib.util
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = None
_CLASS = None


def _load(dataset):
    global _DEVICE, _CLASS
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _DEVICE is None:
        torch.set_num_threads(4)
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        spec = importlib.util.spec_from_file_location('trial_recurrent_model', _HERE / 'recurrent_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.RecurrentMemory
    checkpoint = torch.load(_HERE / (dataset + '_model.pt'), map_location='cpu', weights_only=True)
    model = _CLASS(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(_DEVICE).eval()
    _MODELS[dataset] = model
    return model


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(dataset)
    hist = list(history[-model.maxlen:])
    seq = hist + [0] * (model.maxlen - len(hist))
    seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    return model.predict(seq_t, cand_t)[0].float().cpu().tolist()
"""
    (WORK / "predict.py").write_text(predictor)


def main():
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    write_predictor()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Reserve time for all categories within the nine-hour execution limit.
    global_deadline = time.monotonic() + 8 * 3600
    scores = []
    for index, name in enumerate(DATASETS):
        remaining = max(0.0, global_deadline - time.monotonic())
        deadline = time.monotonic() + remaining / (len(DATASETS) - index)
        scores.append(train_category(name, device, deadline))
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
