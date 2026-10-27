from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
HERE = os.path.dirname(os.path.abspath(__file__))
WORKING = os.path.join(HERE, "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
BATCH = int(os.environ.get("SASREC_BATCH", "512"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "100"))
DIM = int(os.environ.get("SASREC_HIDDEN", "128"))
NEGATIVES = 64
PATIENCE = 30


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
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats.get(name, {}).get("itemnum", 0))
    # Vocabulary bounds only; validation interactions never enter fitting.
    for split in (train, valid):
        for items in split.values():
            if items:
                itemnum = max(itemnum, max(items))
    return train, valid, itemnum


class PrefixDataset(Dataset):
    def __init__(self, train, maxlen):
        self.histories = [
            np.asarray(v, dtype=np.int64) for v in train.values() if len(v) >= 2
        ]
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No training histories with at least two items")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        end = (
            len(history) - 1
            if np.random.random() < 0.5
            else np.random.randint(1, len(history))
        )
        prefix = history[max(0, end - self.maxlen) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-len(prefix) :] = prefix
        return seq, int(history[end])


def worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class CausalBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding dimension must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.attention_dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, allowed, valid):
        batch, length, dim = x.shape
        qkv = (
            self.qkv(self.norm1(x))
            .reshape(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, length, dim)
        x = x + self.residual_dropout(self.projection(out))
        x = x + self.ff(self.norm2(x))
        return x * valid.unsqueeze(-1)


class HistoryModel(nn.Module):
    def __init__(self, itemnum, dim=128, maxlen=100, dropout=0.25, blocks=2, heads=4):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.dim = dim
        self.input_emb = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        # Both embedding roles update the same learned Parameter.
        self.output_emb = self.input_emb
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.position_emb = nn.Embedding(maxlen, dim)
        self.sequence_dropout = nn.Dropout(dropout)
        self.sequence_blocks = nn.ModuleList(
            [CausalBlock(dim, heads, dropout) for _ in range(blocks)]
        )
        self.sequence_norm = nn.LayerNorm(dim)
        # Two candidate-conditioned heads, each with 32-dimensional queries/keys.
        self.key = nn.Linear(dim, 64, bias=False)
        self.query = nn.Linear(dim, 64)
        self.candidate_chunk_size = 16
        self.decay = nn.Parameter(torch.tensor([-2.0, 0.5]))
        self.encoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.empty_context = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.input_emb.weight, std=0.05)
        nn.init.normal_(self.position_emb.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.input_emb.weight[0].zero_()

    def contextualize(self, seq, mask):
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device)
        x = self.input_emb(seq) * math.sqrt(self.dim)
        x = self.sequence_dropout(x + self.position_emb(positions)[None])
        x = x * mask.unsqueeze(-1)
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
        diagonal = torch.eye(length, dtype=torch.bool, device=seq.device)
        allowed = (causal[None] & mask[:, None, :]) | (
            diagonal[None] & ~mask[:, :, None]
        )
        allowed = allowed[:, None]
        for block in self.sequence_blocks:
            x = block(x, allowed, mask)
        return self.sequence_norm(x) * mask.unsqueeze(-1)

    def encode(self, seq):
        mask = seq.ne(0)
        x = self.contextualize(seq, mask)
        # Cache the backbone once; score() reads the history for each candidate.
        return x, mask, x[:, -1]

    def score(self, context, candidates):
        x, mask, last = context
        batch, length, _ = x.shape
        if candidates.shape[1] == 0:
            return x.new_empty((batch, 0))
        keys = self.key(x).reshape(batch, length, 2, 32)
        age = torch.arange(length - 1, -1, -1, device=x.device, dtype=x.dtype)
        penalty = torch.log1p(age)[:, None] * F.softplus(self.decay)[None, :]
        scale = self.log_scale.clamp(math.log(1.0), math.log(40.0)).exp()
        scores = []
        # The same differentiable chunk loop serves training and deployment.
        for start in range(0, candidates.shape[1], self.candidate_chunk_size):
            ids = candidates[:, start : start + self.candidate_chunk_size]
            items = self.output_emb(ids)
            queries = self.query(items).reshape(batch, ids.shape[1], 2, 32)
            logits = torch.einsum("bchk,blhk->bclh", queries, keys) / math.sqrt(32.0)
            logits = logits - penalty[None, None]
            valid = mask[:, None, :, None]
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=2) * valid
            # Empty histories have exactly zero weights and zero summaries.
            weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-8)
            pooled = torch.einsum("bclh,bld->bchd", weights, x).flatten(2)
            features = torch.cat(
                [pooled, last[:, None].expand(-1, ids.shape[1], -1), items],
                dim=-1,
            )
            representation = self.norm(self.encoder(self.dropout(features)))
            representation = torch.where(
                mask.any(1)[:, None, None],
                representation,
                self.empty_context[None, None],
            )
            representation = F.normalize(representation, dim=-1)
            scores.append(
                scale * (items * representation).sum(-1)
                + self.item_bias(ids).squeeze(-1)
            )
        return torch.cat(scores, dim=1)

    def forward(self, seq, candidates):
        return self.score(self.encode(seq), candidates)

    def predict(self, seq, candidates):
        return self.forward(seq, candidates)


def make_validation(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        hist = train[user][-maxlen:]
        seq = [0] * (maxlen - len(hist)) + hist
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError("No validation users")
    return torch.tensor(sequences), torch.tensor(candidates)


@torch.inference_mode()
def evaluate(model, validation, device):
    model.eval()
    seqs, candidates = validation
    hits = 0
    for start in range(0, len(seqs), 128):
        scores = model.predict(
            seqs[start : start + 128].to(device),
            candidates[start : start + 128].to(device),
        )
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        hits += order[:, :10].eq(0).any(1).sum().item()
    return hits / len(seqs)


def check_candidate_readout(model, device):
    # Diagnostics consume runtime, but do not advance training RNG streams or
    # populate parameter .grad fields. Disable dropout for equivalence checks.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = model.training
    chunk_size = model.candidate_chunk_size
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            seq = torch.zeros((3, model.maxlen), dtype=torch.long, device=device)
            seq[1, -1] = 1
            width = min(4, model.maxlen)
            seq[2, -width:] = (
                torch.arange(width, device=device) % model.itemnum + 1
            )
            items = (
                torch.arange(19, device=device)[None].expand(3, -1)
                % model.itemnum + 1
            )
            context = model.encode(seq)
            scores = model.score(context, items)
            assert torch.isfinite(scores).all()
            # Empty and single-item histories must support finite backward passes.
            for row in (0, 1):
                gradients = torch.autograd.grad(
                    model(seq[row : row + 1], items[row : row + 1]).square().mean(),
                    tuple(model.parameters()), allow_unused=True,
                )
                assert all(g is None or torch.isfinite(g).all() for g in gradients)
            gradients = torch.autograd.grad(
                scores.square().mean(), (model.query.weight, model.key.weight),
                retain_graph=True,
            )
            assert all(torch.isfinite(g).all() for g in gradients)
            with torch.no_grad():
                # Independently reconstruct candidate-conditioned attention and
                # scoring, including the learned recency penalty and empty case.
                x, mask, last = context
                keys = model.key(x).reshape(3, model.maxlen, 2, 32)
                embeddings = model.output_emb(items)
                queries = model.query(embeddings).reshape(3, items.shape[1], 2, 32)
                logits = torch.einsum("bchk,blhk->bclh", queries, keys) / math.sqrt(32.0)
                age = torch.arange(model.maxlen - 1, -1, -1, device=device, dtype=x.dtype)
                logits -= torch.log1p(age)[None, None, :, None] * F.softplus(model.decay)
                valid = mask[:, None, :, None]
                weights = logits.masked_fill(~valid, torch.finfo(x.dtype).min).softmax(2) * valid
                weights /= weights.sum(2, keepdim=True).clamp_min(1e-8)
                summaries = torch.einsum("bclh,bld->bchd", weights, x).flatten(2)
                assert torch.equal(summaries[0], torch.zeros_like(summaries[0]))
                features = torch.cat(
                    [summaries, last[:, None].expand(-1, items.shape[1], -1), embeddings], -1
                )
                representation = model.norm(model.encoder(model.dropout(features)))
                representation = torch.where(
                    mask.any(1)[:, None, None], representation, model.empty_context[None, None]
                )
                explicit = (
                    model.log_scale.clamp(math.log(1.0), math.log(40.0)).exp()
                    * (embeddings * F.normalize(representation, dim=-1)).sum(-1)
                    + model.item_bias(items).squeeze(-1)
                )
                torch.testing.assert_close(scores, explicit, rtol=1e-5, atol=1e-5)
                # Changing chunk boundaries and candidate order must preserve scores.
                for size in (1, 7, items.shape[1]):
                    model.candidate_chunk_size = size
                    torch.testing.assert_close(
                        scores, model.score(context, items), rtol=1e-5, atol=1e-5
                    )
                torch.testing.assert_close(
                    scores.flip(1), model.score(context, items.flip(1)),
                    rtol=1e-5, atol=1e-5,
                )
                # Changing a later token cannot affect earlier contextual states.
                if model.maxlen > 1:
                    changed = seq.clone()
                    changed[2, -1] = changed[2, -1] % model.itemnum + 1
                    altered = model.contextualize(changed, changed.ne(0))
                    torch.testing.assert_close(
                        x[:, :-1], altered[:, :-1], rtol=1e-5, atol=1e-5
                    )
    finally:
        model.candidate_chunk_size = chunk_size
        model.train(was_training)


def train_category(name, device, deadline):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    config = dict(
        itemnum=itemnum,
        dim=DIM,
        maxlen=MAXLEN,
        dropout=0.25,
        blocks=2,
        heads=4,
    )
    model = HistoryModel(**config).to(device)
    assert model.input_emb.weight is model.output_emb.weight
    check_candidate_readout(model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.98), weight_decay=1e-5
    )
    assert sum(
        parameter is model.input_emb.weight
        for group in optimizer.param_groups
        for parameter in group["params"]
    ) == 1
    dataset = PrefixDataset(train, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=STEPS * BATCH,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=4,
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
    )
    validation = make_validation(train, valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, batches = 0.0, 0
        for seq, target in loader:
            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            negative = torch.randint(
                1, itemnum + 1, (len(seq), NEGATIVES), device=device
            )
            allowed = negative.ne(target[:, None])
            allowed &= ~negative[:, :, None].eq(seq[:, None, :]).any(-1)
            candidate = torch.cat([target[:, None], negative], dim=1)
            scores = model(seq, candidate)

            # Sole experimental change: independent binary logistic supervision.
            # Add the positive term to the mean over valid negative draws,
            # preserving collision masks and duplicate-negative behavior.
            # Average equally across examples; examples without valid negatives
            # retain their positive term, with no masked-negative contribution.
            positive_loss = F.softplus(-scores[:, 0].float())
            negative_scores = scores[:, 1:].float().masked_fill(~allowed, 0.0)
            negative_loss = F.softplus(negative_scores).masked_fill(~allowed, 0.0).sum(dim=1) / allowed.sum(dim=1).clamp_min(1)
            loss = (positive_loss + negative_loss).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
            if time.monotonic() >= deadline:
                break
        check = (
            epoch == 1
            or epoch % 5 == 0
            or time.monotonic() >= deadline
            or epoch == EPOCHS
        )
        hr = None
        if check:
            hr = evaluate(model, validation, device)
            if hr > best:
                best, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= 15:
                for group in optimizer.param_groups:
                    group["lr"] = max(0.00015, group["lr"] * 0.8)
        suffix = "" if hr is None else f" HR@10={hr:.5f} best={best:.5f}"
        print(
            f"{name} epoch={epoch} loss={loss_sum / max(batches, 1):.5f}{suffix}",
            flush=True,
        )
        if (check and epoch - best_epoch >= PATIENCE) or time.monotonic() >= deadline:
            break
    with open(os.path.join(WORKING, name + "_meta.json"), "w") as f:
        json.dump(config, f)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    assert model.input_emb.weight is model.output_emb.weight
    result = evaluate(model, validation, device)

    # Check the generated predictor against the selected checkpoint using
    # supplied validation histories and candidates, without changing selection.
    import importlib.util

    # Reconstructing a checkpoint initializes parameters before loading them;
    # preserve RNG state around that deployment diagnostic as well.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        spec = importlib.util.spec_from_file_location(
            "history_predictor_verification", os.path.join(WORKING, "predict.py")
        )
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
        loaded = predictor._load(name)
    assert loaded.input_emb.weight is loaded.output_emb.weight
    seqs, candidates = validation
    with torch.inference_mode():
        for index in range(min(3, len(seqs))):
            seq = seqs[index : index + 1].to(device)
            items = candidates[index : index + 1].to(device)
            expected = model.predict(seq, items)[0].cpu()
            # A cached backbone must produce the same candidate-conditioned scores.
            torch.testing.assert_close(
                expected, model.score(model.encode(seq), items)[0].cpu(),
                rtol=1e-5, atol=1e-5,
            )
            history = seqs[index][seqs[index].ne(0)].tolist()
            actual = predictor.predict(
                None, history, candidates[index].tolist(), dataset_name=name
            )
            torch.testing.assert_close(
                torch.tensor(actual, dtype=expected.dtype), expected,
                rtol=1e-5, atol=1e-5,
            )
    del loaded, predictor
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_predictor():
    source = r"""from __future__ import annotations
import importlib.util
import json
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(name):
    global _CLASS
    if name not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _MODELS:
        if _CLASS is None:
            spec = importlib.util.spec_from_file_location(
                'history_training_architecture', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _CLASS = module.HistoryModel
        with open(os.path.join(_HERE, name + '_meta.json')) as f:
            config = json.load(f)
        # Reconstruct the candidate-conditioned architecture from saved config;
        # strict loading below requires every attention and scoring parameter.
        model = _CLASS(**config).to(_DEVICE)
        assert model.key.out_features == 64 and model.query.out_features == 64
        assert model.candidate_chunk_size == 16
        state = torch.load(os.path.join(_HERE, name + '_model.pt'),
                           map_location=_DEVICE, weights_only=True)
        assert model.input_emb.weight is model.output_emb.weight
        model.load_state_dict(state, strict=True)
        assert model.input_emb.weight is model.output_emb.weight
        model.eval()
        _MODELS[name] = model
    return _MODELS[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    model = _load(dataset_name if dataset_name is not None else _NAMES[0])
    history = list(history)[-model.maxlen:]
    seq = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    items = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return model.predict(seq, items)[0].cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(source)


def main():
    os.makedirs(WORKING, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Validation: supplied holdout, seed=42, up to 200 users/category, "
        "99 unique uniform negatives excluding training and validation items; "
        "HR@10 = fraction of users whose target ranks in the first 10; "
        "category results are macro-averaged. No score postprocessing. "
        "Training objective: independent binary logistic supervision, positive "
        "softplus plus mean softplus over valid negatives from 64 uniform draws; "
        "existing collision masks and duplicates preserved, averaged equally "
        "across examples; examples without valid negatives retain the positive term.",
        flush=True,
    )
    total_deadline = time.monotonic() + 8 * 60 * 60
    write_predictor()
    results = []
    for index, name in enumerate(DATASETS):
        remaining = len(DATASETS) - index
        now = time.monotonic()
        category_deadline = now + max(1.0, total_deadline - now) / remaining
        results.append(train_category(name, device, category_deadline))
    score = sum(results) / len(results)
    with open(os.path.join(WORKING, "validation_metrics.json"), "w") as f:
        json.dump(
            {
                "metric": "HR@10",
                "evaluation_seed": 42,
                "users_per_category": 200,
                "unique_negatives": 99,
                "per_category": dict(zip(DATASETS, results)),
                "macro_average": score,
                "checkpoint_policy": "one best validation checkpoint per category",
                "training_objective": "independent binary logistic supervision: softplus(-positive_score) plus mean softplus over valid negatives from 64 uniform draws, existing collision masks and duplicates preserved, equal-weight batch mean; examples without valid negatives retain the positive term",
            },
            f,
            indent=2,
        )
    print(f"Final Validation Score: {score}")
    for name, hr in zip(DATASETS, results):
        print(f"{name} selected-checkpoint validation HR@10={hr:.5f}")
    print(
        f"Local macro validation HR@10={score:.5f}; "
        f"delta versus 0.4775={score - 0.4775:+.5f}; "
        f"delta versus baseline 0.505={score - 0.505:+.5f}. "
        "Pinned host validation is authoritative; hidden test data is not used.",
        flush=True,
    )


if __name__ == "__main__":
    main()
