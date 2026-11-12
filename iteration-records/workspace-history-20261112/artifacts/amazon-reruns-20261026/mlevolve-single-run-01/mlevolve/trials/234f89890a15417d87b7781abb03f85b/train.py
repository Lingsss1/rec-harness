from __future__ import annotations

import inspect
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
ROOT = os.path.dirname(os.path.abspath(__file__))
WORKING = os.path.join(ROOT, "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "64"))
DIM = int(os.environ.get("GRU_DIM", "128"))
NEGATIVES = 1024
MAX_POSITIONS = 2048
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOTAL_SECONDS = 8.5 * 3600


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    values = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                values[int(fields[0])].append(int(fields[1]))
    return dict(values)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(path):
        with open(path) as f:
            itemnum = int(json.load(f)[name]["itemnum"])
    else:
        itemnum = max(i for h in train.values() for i in h)
    return train, valid, itemnum


class HistoryDataset(Dataset):
    def __init__(self, train, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No training histories with next-item targets")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        h = self.histories[index]
        end = len(h)
        if len(h) > 3 and random.random() < 0.35:
            end = random.randint(2, len(h))
        window = h[max(0, end - self.maxlen - 1) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        target = np.zeros(self.maxlen, dtype=np.int64)
        length = len(window) - 1
        seq[:length] = window[:-1]
        target[:length] = window[1:]
        return seq, target


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class GRURec(nn.Module):
    def __init__(self, itemnum, dim=128, layers=2, dropout=0.2, maxlen=64):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.embedding = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            dim,
            dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.context_output = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.constant_(self.gate.bias, -2.0)
        self.norm = nn.LayerNorm(dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, dim, bias=False)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        for name, parameter in self.gru.named_parameters():
            if "weight_hh" in name:
                for block in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(block)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, seq):
        item_memory = self.embedding(seq)
        embedded = self.input_dropout(item_memory)
        features, _ = self.gru(embedded)
        queries = self.query(features)
        # Atomic change: retrieve purchase identities directly, rather than
        # retrieving recurrently compressed states. These embeddings and all
        # retrieval projections train jointly with the recurrent network.
        keys = self.key(item_memory)
        values = self.value(item_memory)
        valid = seq.ne(0)
        length = seq.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
        allowed = causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
        scores = (queries @ keys.transpose(1, 2)) * (features.shape[-1] ** -0.5)
        scores = scores.masked_fill(~allowed, float("-inf"))
        scores = scores.masked_fill(~valid.unsqueeze(-1), 0.0)
        weights = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0.0)
        context = self.context_output(weights @ values)
        context = context.masked_fill(~valid.unsqueeze(-1), 0.0)
        gate = torch.sigmoid(self.gate(torch.cat([features, context], dim=-1)))
        features = features + gate * context
        features = self.output_dropout(self.norm(features))
        return self.output(features)

    def score_features(self, features, candidates):
        embeddings = self.embedding(candidates)
        bias = self.item_bias(candidates).squeeze(-1)
        return (features.unsqueeze(-2) * embeddings).sum(-1) + bias

    def predict(self, seq, candidates):
        features = self.forward(seq)
        last = seq.ne(0).sum(1).clamp(min=1) - 1
        features = features[torch.arange(seq.shape[0], device=seq.device), last]
        return self.score_features(features, candidates)


def validation_cases(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        hist = train[user][-maxlen:]
        seq = np.zeros(maxlen, dtype=np.int64)
        seq[: len(hist)] = hist
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Insufficient distinct evaluation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError("No eligible validation users")
    return (
        torch.as_tensor(np.asarray(sequences), dtype=torch.long),
        torch.as_tensor(np.asarray(candidates), dtype=torch.long),
    )


@torch.inference_mode()
def evaluate(model, cases):
    model.eval()
    hits, ndcg = 0, 0.0
    sequences, candidates = cases
    for start in range(0, len(sequences), 64):
        seq = sequences[start : start + 64].to(DEVICE)
        cand = candidates[start : start + 64].to(DEVICE)
        scores = model.predict(seq, cand).float().cpu().numpy()
        ranks = np.argsort(-scores, axis=1, kind="stable").argmin(axis=1)
        hits += int((ranks < 10).sum())
        ndcg += float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0).sum())
    return hits / len(sequences), ndcg / len(sequences)


def ranking_loss(model, seq, target):
    encoded = model(seq)
    rows, columns = target.nonzero(as_tuple=True)
    if len(rows) > MAX_POSITIONS:
        chosen = torch.randperm(len(rows), device=seq.device)[:MAX_POSITIONS]
        rows, columns = rows[chosen], columns[chosen]
    features = encoded[rows, columns]
    positives = target[rows, columns]
    negatives = torch.randint(1, model.itemnum + 1, (NEGATIVES,), device=seq.device)
    positive_scores = (features * model.embedding(positives)).sum(-1)
    positive_scores = positive_scores + model.item_bias(positives).squeeze(-1)
    negative_scores = features @ model.embedding(negatives).T
    negative_scores = negative_scores + model.item_bias(negatives).squeeze(-1)
    seen = (
        (seq.unsqueeze(-1) == negatives.view(1, 1, -1))
        | (target.unsqueeze(-1) == negatives.view(1, 1, -1))
    ).any(dim=1)
    invalid = seen[rows]
    usable = (~invalid).any(dim=1)
    positive_scores = positive_scores[usable].float()
    negative_scores = negative_scores[usable].float()
    invalid = invalid[usable]
    if positive_scores.numel() == 0:
        return encoded.sum() * 0.0
    log_weights = torch.nn.functional.log_softmax(
        negative_scores.masked_fill(invalid, float("-inf")), dim=1
    )
    negative_scores = negative_scores.masked_fill(invalid, 0.0)
    margins = positive_scores[:, None] - negative_scores
    log_pairs = log_weights + torch.nn.functional.logsigmoid(margins)
    log_pairs = log_pairs.masked_fill(invalid, float("-inf"))
    losses = -torch.logsumexp(log_pairs, dim=1)
    penalty = (log_weights.exp() * negative_scores.square()).sum(dim=1)
    return (losses + 0.001 * penalty).mean()


def check_ranking_loss():
    from unittest.mock import patch

    class FixedScores:
        itemnum = 4

        def __init__(self, values, length):
            self.scores = torch.tensor(values, dtype=torch.float32, requires_grad=True)
            self.encoded = torch.ones((1, length, 1), requires_grad=True)

        def __call__(self, seq):
            return self.encoded

        def embedding(self, items):
            return self.scores[items].unsqueeze(-1)

        def item_bias(self, items):
            return torch.zeros((*items.shape, 1), dtype=torch.float32)

    def run(values, draws, seq_values=(2,), target_values=(1,)):
        model = FixedScores(values, len(seq_values))
        seq = torch.tensor([seq_values], dtype=torch.long)
        target = torch.tensor([target_values], dtype=torch.long)
        negatives = torch.tensor(draws, dtype=torch.long)
        negatives = negatives.repeat((NEGATIVES + len(draws) - 1) // len(draws))
        with patch.object(torch, "randint", return_value=negatives[:NEGATIVES]):
            loss = ranking_loss(model, seq, target)
        assert torch.isfinite(loss).item()
        gradients = torch.autograd.grad(
            loss, (model.encoded, model.scores), allow_unused=True
        )
        for gradient in gradients:
            if gradient is not None:
                assert torch.isfinite(gradient).all().item()
        return loss.detach(), gradients

    values = [0.0, 0.75, 7.0, -0.25, 1.25]
    single, _ = run(values, [3])
    expected = torch.nn.functional.softplus(torch.tensor(-1.0)) + 0.001 * 0.25**2
    torch.testing.assert_close(single, expected)
    mixed, gradients = run(values, [1, 2, 3])
    torch.testing.assert_close(mixed, single)
    assert gradients[1][2].item() == 0.0
    changed = list(values)
    changed[2] = -10000.0
    changed_loss, _ = run(changed, [1, 2, 3])
    torch.testing.assert_close(changed_loss, mixed)
    masked, gradients = run(values, [1, 2])
    assert masked.item() == 0.0
    for gradient in gradients:
        assert gradient is None or gradient.eq(0).all().item()
    filtered, gradients = run(values, [3], (2, 2), (1, 1))
    torch.testing.assert_close(filtered, single)
    torch.testing.assert_close(gradients[0][0, 0], gradients[0][0, 1])
    run([0.0, -10000.0, 10000.0, 10000.0, -10000.0], [2, 3, 4])
    run([0.0, 10000.0, -10000.0, -10000.0, 10000.0], [2, 3, 4])


def train_category(name, allowance):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    config = dict(itemnum=itemnum, dim=DIM, layers=2, dropout=0.2, maxlen=MAXLEN)
    model = GRURec(**config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=1e-5, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )
    dataset = HistoryDataset(train, MAXLEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=DEVICE.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42),
    )
    iterator = iter(loader)
    cases = validation_cases(train, valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, completed = 0.0, 0
        for step in range(STEPS):
            try:
                seq, target = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                seq, target = next(iterator)
            seq = seq.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = ranking_loss(model, seq, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite ranking loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            completed += 1
            if time.monotonic() - started >= allowance:
                break
        expired = time.monotonic() - started >= allowance
        stop = False
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired:
            hr, ndcg = evaluate(model, cases)
            scheduler.step(hr)
            if (hr, ndcg) > (best_hr, best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(
                    {"config": config, "state_dict": model.state_dict()}, checkpoint
                )
            print(
                f"{name} epoch={epoch} loss={loss_sum/max(completed,1):.5f} "
                f"HR@10={hr:.5f} NDCG@10={ndcg:.5f}",
                flush=True,
            )
            stop = epoch - best_epoch >= 35
        else:
            print(
                f"{name} epoch={epoch} loss={loss_sum/max(completed,1):.5f}", flush=True
            )
        if expired or stop:
            break
    saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    final_hr, _ = evaluate(model, cases)
    with open(os.path.join(WORKING, name + "_validation.json"), "w") as f:
        json.dump(
            {
                "HR@10": final_hr,
                "selected_epoch": best_epoch,
                "users": len(cases[0]),
                "negative_candidates": 99,
                "evaluation_seed": 42,
                "training_seed": 42,
                "splits": "provided train and valid files; unchanged",
                "target": "first validation purchase",
                "metric": "fraction of targets ranked in first 10 of 100 candidates",
                "architecture": "GRU with gated causal item-memory attention",
            },
            f,
        )
    del iterator, loader, model, optimizer, scheduler, saved
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return final_hr


def write_predict_script():
    prefix = """from __future__ import annotations
import os
import torch
from torch import nn

"""
    suffix = """
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _MODELS:
        checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'),
                                map_location=_DEVICE, weights_only=True)
        model = GRURec(**checkpoint['config']).to(_DEVICE)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        _MODELS[name] = model
    return _MODELS[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    model = _load(dataset_name)
    if len(candidates) == 0:
        return []
    hist = list(history[-model.maxlen:])
    seq = [0] * model.maxlen
    seq[:len(hist)] = hist
    seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    return model.predict(seq_t, cand_t)[0].float().cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(prefix + inspect.getsource(GRURec) + "\n" + suffix)


def main():
    os.makedirs(WORKING, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    check_ranking_loss()
    write_predict_script()
    started = time.monotonic()
    scores = []
    for index, name in enumerate(DATASETS):
        remaining = TOTAL_SECONDS - (time.monotonic() - started)
        allowance = max(1.0, remaining / (len(DATASETS) - index))
        scores.append(train_category(name, allowance))
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
