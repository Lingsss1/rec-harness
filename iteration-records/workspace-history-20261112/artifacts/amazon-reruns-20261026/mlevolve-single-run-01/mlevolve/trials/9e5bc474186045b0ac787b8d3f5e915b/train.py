from __future__ import annotations

import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "64"))
HIDDEN = int(os.environ.get("SASREC_HIDDEN", "96"))
BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
HEADS = int(os.environ.get("SASREC_HEADS", "2"))
DROPOUT = float(os.environ.get("SASREC_DROPOUT", "0.25"))
LR = float(os.environ.get("SASREC_LR", "0.001"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))
NEGATIVES = 512
RELATIVE_CLIP = 32


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    if name in stats:
        itemnum = int(stats[name]["itemnum"])
    else:
        itemnum = max(i for h in train.values() for i in h)
        itemnum = max(itemnum, max((i for h in valid.values() for i in h), default=0))
    return train, valid, itemnum


class MaskedHistories(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.itemnum = itemnum
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No training histories with at least two items")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        end = len(history)
        if np.random.random() < 0.5:
            end = np.random.randint(2, len(history) + 1)
        tokens = history[max(0, end - self.maxlen) : end].copy()
        length = len(tokens)
        masked = np.zeros(length, dtype=bool)
        masked[:-1] = (tokens[:-1] != 0) & (np.random.random(length - 1) < 0.15)
        masked[-1] = True
        targets = np.zeros(self.maxlen, dtype=np.int64)
        seq = np.zeros(self.maxlen, dtype=np.int64)
        targets[-length:] = np.where(masked, tokens, 0)
        tokens[masked] = self.itemnum + 1
        seq[-length:] = tokens
        return torch.from_numpy(seq), torch.from_numpy(targets)


def init_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BidirectionalBlock(nn.Module):
    def __init__(self, hidden, heads, dropout, relative_clip=32):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.relative_clip = relative_clip
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)
        # One jointly trained signed-distance bias per attention head.
        # Offsets outside the radius share the corresponding boundary bucket.
        self.relative_bias = nn.Embedding(2 * relative_clip + 1, heads)

    def forward(self, x, allowed, active, relative_indices):
        b, length, hidden = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            b, length, 3, self.heads, hidden // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_bias(relative_indices)
        bias = bias.permute(2, 0, 1).unsqueeze(0).to(q.dtype)
        # Additive masking retains padding exclusion while allowing learned
        # relative distances to affect the attention logits.
        attention_bias = bias.expand(b, -1, -1, -1).masked_fill(~allowed, float("-inf"))
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(b, length, hidden)
        x = x + self.residual_dropout(self.projection(attention))
        x = x + self.ffn(self.norm2(x))
        return x * active


def _supervised_positions(targets):
    selected = targets.ne(0).flatten().nonzero(as_tuple=False).squeeze(1)
    endpoints = selected[(selected + 1).remainder(targets.shape[1]).eq(0)]
    if endpoints.numel() > 768:
        return endpoints[torch.randperm(endpoints.numel(), device=targets.device)[:768]]
    if selected.numel() > 768:
        interior = selected[(selected + 1).remainder(targets.shape[1]).ne(0)]
        interior = interior[
            torch.randperm(interior.numel(), device=targets.device)[
                : 768 - endpoints.numel()
            ]
        ]
        selected = torch.cat([endpoints, interior])
    return selected


class MaskedRec(nn.Module):
    def __init__(
        self,
        itemnum,
        hidden=96,
        blocks=2,
        heads=2,
        dropout=0.25,
        maxlen=64,
        relative_clip=32,
    ):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if maxlen < 1 or relative_clip < 1:
            raise ValueError("Invalid sequence length or relative distance radius")
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.hidden = hidden
        self.item_emb = nn.Embedding(itemnum + 2, hidden, padding_idx=0)
        self.position_emb = nn.Embedding(maxlen, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                BidirectionalBlock(hidden, heads, dropout, relative_clip)
                for _ in range(blocks)
            ]
        )
        self.output_transform = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.item_bias = nn.Embedding(itemnum + 2, 1, padding_idx=0)
        self.log_scale = nn.Parameter(torch.tensor(math.log(12.0)))
        positions = torch.arange(maxlen)
        # Positive offsets denote keys preceding the query.
        relative_indices = (positions[:, None] - positions[None, :]).clamp(
            -relative_clip, relative_clip
        ) + relative_clip
        self.register_buffer("relative_indices", relative_indices, persistent=False)
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.item_bias.weight.zero_()
            for block in self.blocks:
                block.relative_bias.weight.zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        active = seq.ne(0)
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device)
        x = (
            self.item_emb(seq) * math.sqrt(self.hidden)
            + self.position_emb(positions)[None]
        )
        x = self.input_dropout(self.input_norm(x)) * active.unsqueeze(-1)
        allowed = active[:, None, None, :]
        relative_indices = self.relative_indices[:length, :length]
        for block in self.blocks:
            x = block(x, allowed, active.unsqueeze(-1), relative_indices)
        return F.normalize(self.output_transform(x), dim=-1)

    def score(self, features, candidates):
        embeddings = F.normalize(self.item_emb(candidates), dim=-1)
        scale = self.log_scale.exp().clamp(1.0, 40.0)
        if candidates.ndim == 1:
            return scale * (features @ embeddings.T) + self.item_bias(
                candidates
            ).squeeze(-1)
        return scale * (features.unsqueeze(1) * embeddings).sum(-1) + self.item_bias(
            candidates
        ).squeeze(-1)

    def training_loss(self, seq, targets):
        selected = _supervised_positions(targets)
        features = self.encode(seq).reshape(-1, self.hidden)[selected]
        positive_ids = targets.flatten()[selected]
        positive_emb = F.normalize(self.item_emb(positive_ids), dim=-1)
        scale = self.log_scale.exp().clamp(1.0, 40.0)
        positive = scale * (features * positive_emb).sum(-1) + self.item_bias(
            positive_ids
        ).squeeze(-1)
        negative_ids = torch.randint(
            1, self.itemnum + 1, (NEGATIVES,), device=seq.device
        )
        negative = self.score(features, negative_ids)
        negative = negative.masked_fill(
            positive_ids[:, None].eq(negative_ids[None]), -1e4
        )
        logits = torch.cat([positive[:, None], negative], dim=1)
        return F.cross_entropy(
            logits.float(),
            torch.zeros(len(selected), dtype=torch.long, device=seq.device),
        )

    def predict(self, seq, candidates):
        return self.score(self.encode(seq)[:, -1], candidates)


def inference_sequence(history, itemnum, maxlen):
    context = list(history[-(maxlen - 1) :]) if maxlen > 1 else []
    seq = [0] * maxlen
    seq[-len(context) - 1 :] = context + [itemnum + 1]
    return seq


def validation_data(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            candidate = rng.randint(1, itemnum)
            if candidate not in excluded:
                negatives.add(candidate)
        sequences.append(inference_sequence(train[user], itemnum, maxlen))
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError("No eligible validation users")
    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(candidates, dtype=torch.long),
    )


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    seq, candidates = validation
    hits, ndcg = 0, 0.0
    for start in range(0, len(seq), 64):
        scores = (
            model.predict(
                seq[start : start + 64].to(device),
                candidates[start : start + 64].to(device),
            )
            .float()
            .cpu()
        )
        ranks = (
            torch.argsort(scores, dim=1, descending=True, stable=True)
            .eq(0)
            .long()
            .argmax(dim=1)
        )
        hits += ranks.lt(10).sum().item()
        ndcg += (
            torch.where(ranks < 10, 1.0 / torch.log2(ranks.float() + 2), 0.0)
            .sum()
            .item()
        )
    return hits / len(seq), ndcg / len(seq)


def verify_training(model, device):
    # Validate the actual category network without fitting another model,
    # advancing training RNG streams, or retaining diagnostic gradients.
    numpy_state = np.random.get_state()
    was_training = model.training
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            np.random.seed(123)
            torch.random.default_generator.manual_seed(123)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(123)
            history = [1, 2, 3, 4, 5]
            saw_interior = False
            for maxlen in (3, 8):
                dataset = MaskedHistories({1: history}, 5, maxlen)
                for _ in range(32):
                    state = np.random.get_state()
                    end = len(history)
                    if np.random.random() < 0.5:
                        end = np.random.randint(2, len(history) + 1)
                    np.random.set_state(state)
                    seq, targets = dataset[0]
                    original = torch.tensor(history[max(0, end - maxlen) : end])
                    length = len(original)
                    masked = targets.ne(0)
                    assert seq[-1].item() == 6
                    assert targets[-1].item() == history[end - 1]
                    assert torch.equal(seq.eq(6), masked)
                    assert not targets[:-length].any().item()
                    assert not seq[:-length].any().item()
                    assert torch.equal(
                        torch.where(masked, targets, seq)[-length:], original
                    )
                    saw_interior |= masked[:-1].any().item()
            assert saw_interior
            for rows, columns in ((2, 4), (4, 300), (800, 2)):
                targets = torch.ones(rows, columns, dtype=torch.long, device=device)
                selected = _supervised_positions(targets)
                assert selected.numel() == min(768, targets.numel())
                assert selected.unique().numel() == selected.numel()
                endpoints = torch.arange(
                    columns - 1, rows * columns, columns, device=device
                )
                if rows <= 768:
                    assert torch.isin(endpoints, selected).all().item()
                else:
                    assert torch.isin(selected, endpoints).all().item()
            seq = torch.zeros(2, model.maxlen, dtype=torch.long, device=device)
            targets = torch.zeros_like(seq)
            seq[:, -1] = model.itemnum + 1
            targets[:, -1] = 1
            if model.maxlen > 1:
                seq[0, -2] = model.itemnum + 1
                targets[0, -2] = min(2, model.itemnum)
            model.train()
            loss = model.training_loss(seq, targets)
            assert torch.isfinite(loss).item()
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients
            assert all(torch.isfinite(g).all().item() for g in gradients)
            for block in model.blocks:
                gradient = block.relative_bias.weight.grad
                assert gradient is not None
                assert torch.isfinite(gradient).all().item()
    finally:
        np.random.set_state(numpy_state)
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def train_category(name, device, deadline):
    seed_all(42)
    train, valid, itemnum = load_data(name)
    config = dict(
        itemnum=itemnum,
        hidden=HIDDEN,
        blocks=BLOCKS,
        heads=HEADS,
        dropout=DROPOUT,
        maxlen=MAXLEN,
        relative_clip=RELATIVE_CLIP,
    )
    model = MaskedRec(**config).to(device)
    verify_training(model, device)
    dataset = MaskedHistories(train, itemnum, MAXLEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        worker_init_fn=init_worker,
        generator=torch.Generator().manual_seed(42),
        drop_last=False,
    )
    iterator = iter(loader)
    validation = validation_data(train, valid, itemnum, MAXLEN)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=0.01
    )
    best_key = (-1.0, -1.0)
    best_epoch = 0
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best_state = None
    steps_done = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_total, completed = 0.0, 0
        for _ in range(STEPS):
            try:
                seq, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                seq, targets = next(iterator)
            seq = seq.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            warmup = min(1.0, (steps_done + 1) / 300.0)
            for group in optimizer.param_groups:
                group["lr"] = LR * warmup
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(seq, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            steps_done += 1
            loss_total += float(loss.detach())
            completed += 1
            if time.monotonic() >= deadline and steps_done * BATCH >= len(dataset):
                break
        should_evaluate = (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
            or time.monotonic() >= deadline
        )
        message = f"{name} epoch={epoch} loss={loss_total / max(completed, 1):.5f}"
        if should_evaluate:
            hr, ndcg = evaluate(model, validation, device)
            message += f" validation_HR@10={hr:.5f} validation_NDCG@10={ndcg:.5f}"
            if (hr, ndcg) > best_key:
                best_key = (hr, ndcg)
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(best_state, checkpoint)
        print(message, flush=True)
        if steps_done * BATCH >= len(dataset):
            if time.monotonic() >= deadline or (
                should_evaluate and epoch - best_epoch >= PATIENCE
            ):
                break
    if best_state is None:
        raise RuntimeError("Training produced no selected checkpoint")
    model.load_state_dict(best_state)
    final_hr, _ = evaluate(model, validation, device)
    with open(os.path.join(WORKING, name + "_meta.pkl"), "wb") as f:
        pickle.dump(config, f)
    del iterator, loader, model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_hr


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_METAS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODULE = None


def _load(dataset):
    global _MODULE
    if dataset in _MODELS:
        return
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location(
            'relative_maskedrec_trial_model', os.path.join(_HERE, '..', 'train.py')
        )
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        config = pickle.load(f)
    model = _MODULE.MaskedRec(**config).to(_DEVICE)
    state = torch.load(
        os.path.join(_HERE, dataset + '_model.pt'),
        map_location=_DEVICE, weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    _MODELS[dataset] = model
    _METAS[dataset] = config


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = _NAMES[0] if dataset_name is None else dataset_name
    if dataset not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(dataset))
    _load(dataset)
    config = _METAS[dataset]
    seq = _MODULE.inference_sequence(history, config['itemnum'], config['maxlen'])
    sequence = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    candidate_tensor = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return _MODELS[dataset].predict(sequence, candidate_tensor)[0].float().cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(script)


def main():
    seed_all(42)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Validation: supplied validation split; up to 200 users/category; "
        "99 unique uniform negatives/user; evaluation seed=42; "
        "HR@10=mean(target rank < 10); four-category macro average. "
        "Training uses only supplied training histories; "
        "one jointly trained relative-attention model per category.",
        flush=True,
    )
    finish_by = time.monotonic() + 8.5 * 3600
    scores = []
    for index, name in enumerate(DATASETS):
        remaining = len(DATASETS) - index
        deadline = time.monotonic() + max(0.0, finish_by - time.monotonic()) / remaining
        scores.append(train_category(name, device, deadline))
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
