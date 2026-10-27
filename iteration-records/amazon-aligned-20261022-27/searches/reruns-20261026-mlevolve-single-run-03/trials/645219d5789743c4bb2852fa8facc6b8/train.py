from __future__ import annotations

import inspect
import json
import math
import os
import random
import time
import traceback
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
        self.output_norm = nn.LayerNorm(hidden)
        # One jointly trained candidate-conditioned scorer replaces the readout.
        self.scorer = nn.Sequential(
            nn.Linear(5 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
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
        x = self.input_norm(self.embedding(seq)) * mask
        recurrent, _ = self.gru(self.dropout(x))
        return self.dropout(self.output_norm(recurrent)) * mask

    def _score_chunk(self, states, keys, values, candidate_embeddings, allowed):
        # Each candidate queries contextual GRU states independently. The mask
        # includes observed positions through t, never the next-item target.
        queries = self.query_projection(candidate_embeddings)
        logits = torch.einsum("bpch,blh->bpcl", queries, keys)
        logits = logits / math.sqrt(queries.shape[-1])
        mask = allowed.unsqueeze(2)
        logits = logits.masked_fill(~mask, float("-inf"))
        logits = logits.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)
        attention = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
        context = torch.einsum("bpcl,blh->bpch", attention, values)
        states = states.unsqueeze(2).expand_as(candidate_embeddings)
        features = torch.cat([
            context, states, candidate_embeddings,
            context * candidate_embeddings, states * candidate_embeddings,
        ], dim=-1)
        return self.scorer(features).squeeze(-1)

    def _score(self, h, seq, candidates, positions,
               position_chunk=8, candidate_chunk=16):
        # candidates: [batch, supervised positions, candidates per position].
        # Checkpoint each chunk so backward recomputes its attention/MLP
        # activations. Concatenation preserves gradients and loss normalization.
        from torch.utils.checkpoint import checkpoint

        if candidates.shape[-1] == 0:
            return h.new_empty(candidates.shape)
        keys = self.key_projection(h)
        values = self.value_projection(h)
        history_indices = torch.arange(seq.shape[1], device=seq.device)
        position_scores = []
        for start in range(0, positions.shape[1], position_chunk):
            stop = start + position_chunk
            selected = positions[:, start:stop]
            states = h.gather(
                1, selected.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            )
            allowed = (
                (history_indices[None, None, :] <= selected.unsqueeze(-1))
                & seq.ne(0).unsqueeze(1)
                & seq.gather(1, selected).ne(0).unsqueeze(-1)
            )
            candidate_scores = []
            for offset in range(0, candidates.shape[-1], candidate_chunk):
                ids = candidates[:, start:stop, offset:offset + candidate_chunk]
                embeddings = self.embedding(ids)
                if self.training and torch.is_grad_enabled():
                    scores = checkpoint(
                        self._score_chunk, states, keys, values, embeddings,
                        allowed, use_reentrant=False,
                    )
                else:
                    scores = self._score_chunk(
                        states, keys, values, embeddings, allowed
                    )
                candidate_scores.append(scores + self.item_bias(ids).squeeze(-1))
            position_scores.append(torch.cat(candidate_scores, dim=-1))
        return torch.cat(position_scores, dim=1)

    def forward(self, seq, positives, negatives):
        h = self.encode(seq)
        positions = torch.arange(seq.shape[1], device=seq.device)
        positions = positions.unsqueeze(0).expand(seq.shape[0], -1)
        # Share the exact scoring path for positives and the reused negatives.
        candidates = torch.cat([
            positives.unsqueeze(-1),
            negatives.unsqueeze(1).expand(-1, seq.shape[1], -1),
        ], dim=-1)
        scores = self._score(h, seq, candidates, positions)
        return scores[:, :, 0], scores[:, :, 1:]

    def predict(self, seq, candidates):
        h = self.encode(seq)
        lengths = seq.ne(0).sum(1).clamp_min(1)
        positions = (lengths - 1).unsqueeze(1)
        return self._score(h, seq, candidates.unsqueeze(1), positions)[:, 0]


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


@torch.inference_mode()
def verify_scorer(model, validation, saved, device):
    # Run bounded checks on supplied validation inputs only, after training.
    # No metrics or checkpoint-selection rules are changed by these checks.
    model.eval()
    sequences, candidates = validation
    seq = torch.as_tensor(sequences[:2, :8], device=device)
    cand = torch.as_tensor(candidates[:2, :5], device=device)
    seq = torch.cat([seq, torch.zeros_like(seq[:1])], dim=0)
    cand = torch.cat([cand, cand[:1]], dim=0)
    reference = model.predict(seq, cand)
    assert torch.isfinite(reference).all(), "Nonfinite padded-history scores"

    def check(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)

    permutation = torch.arange(cand.shape[1] - 1, -1, -1, device=device)
    check(model.predict(seq, cand[:, permutation]), reference[:, permutation])
    h = model.encode(seq)
    positions = torch.arange(seq.shape[1], device=device)
    positions = positions.unsqueeze(0).expand(seq.shape[0], -1)
    expanded = cand.unsqueeze(1).expand(-1, seq.shape[1], -1)
    scores = model._score(h, seq, expanded, positions)
    check(model._score(
        h, seq, expanded, positions, position_chunk=3, candidate_chunk=2
    ), scores)
    positives = cand[:, :1].expand(-1, seq.shape[1])
    pos_scores, neg_scores = model(seq, positives, cand)
    check(pos_scores, scores[:, :, 0])
    check(neg_scores, scores)
    final = seq.ne(0).sum(1).clamp_min(1) - 1
    rows = torch.arange(seq.shape[0], device=device)
    check(scores[rows, final], reference)
    # Removing future observations must preserve every earlier prefix score.
    for length in (1, 3, seq.shape[1]):
        prefix = seq[:, :length]
        prefix_final = prefix.ne(0).sum(1).clamp_min(1) - 1
        check(model.predict(prefix, cand), scores[rows, prefix_final])
    model.load_state_dict(saved["state_dict"], strict=True)
    check(model.predict(seq, cand), reference)


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
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"{name}: nonfinite loss at epoch={epoch}, batch={batches + 1}; "
                    f"ranking={ranking_loss.detach().item()}, "
                    f"positive={positive_loss.detach().item()}, "
                    f"negative={negative_loss.detach().item()}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            # Complete at least one update, then honor the allocation within
            # the epoch. Validation and checkpoint saving still run below.
            if time.monotonic() >= deadline:
                break
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
    model.load_state_dict(saved["state_dict"], strict=True)
    verify_scorer(model, validation, saved, device)
    final_hr, _ = evaluate(model, validation, device)
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_hr


def write_predictor():
    # Export the complete shared scorer and contextual encoder. The checkpoint
    # import is local to _score, so inspect.getsource includes that dependency.
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
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        spec = importlib.util.spec_from_file_location('trial_recurrent_model', _HERE / 'recurrent_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.RecurrentMemory
    checkpoint = torch.load(_HERE / (dataset + '_model.pt'), map_location='cpu', weights_only=True)
    model = _CLASS(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
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
        # Keep validation, scorer verification, and exported inference consistent.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
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
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        print(failure, flush=True)
        try:
            WORK.mkdir(parents=True, exist_ok=True)
            (WORK / "failure_traceback.txt").write_text(failure)
        except Exception:
            traceback.print_exc()
        raise
