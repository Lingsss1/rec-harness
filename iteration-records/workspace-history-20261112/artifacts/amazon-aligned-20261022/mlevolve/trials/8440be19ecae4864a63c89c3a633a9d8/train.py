from __future__ import annotations

import copy
import importlib.util
import json
import math
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
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("NAIS_BATCH", "256"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
DIM = int(os.environ.get("NAIS_DIM", "96"))
NEGATIVES = 16
PATIENCE = 24
SECONDS_PER_CATEGORY = float(os.environ.get("NAIS_SECONDS_PER_CATEGORY", "1800"))


def seed_everything(seed=42):
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
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats[name]["itemnum"]) if name in stats else 0
    else:
        itemnum = 0
    itemnum = max(
        itemnum,
        max((max(h) for h in train.values() if h), default=0),
        max((max(h) for h in valid.values() if h), default=0),
    )
    return train, valid, itemnum


class PurchaseDataset(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64)
            for h in train.values()
            if len(set(h)) >= 2 and len(set(h)) < itemnum
        ]
        self.known = [set(h.tolist()) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        # Predict a future purchase using only its preceding interactions.
        t = np.random.randint(1, len(history))
        target = int(history[t])
        context = history[:t]
        if len(context) > self.maxlen:
            # Mix full-history coverage and the recent window during learning.
            if np.random.random() < 0.5:
                context = context[-self.maxlen :]
            else:
                indices = np.sort(
                    np.random.choice(len(context), self.maxlen, replace=False)
                )
                context = context[indices]
        if len(context) > 1:
            keep = np.random.random(len(context)) >= 0.15
            if not keep.any():
                keep[np.random.randint(len(context))] = True
            context = context[keep]
        sequence = np.zeros(self.maxlen, dtype=np.int64)
        sequence[-len(context) :] = context
        negatives = []
        known = self.known[index]
        while len(negatives) < NEGATIVES:
            item = np.random.randint(1, self.itemnum + 1)
            if item not in known:
                negatives.append(item)
        candidates = np.asarray([target] + negatives, dtype=np.int64)
        return sequence, candidates


def check_purchase_dataset():
    # Exercise prefix boundaries, repeats, dropout, and batch shapes without
    # advancing training RNG state. Architecture and inference stay unchanged.
    from unittest.mock import patch

    state = np.random.get_state()
    try:
        dataset = PurchaseDataset({1: [1, 2, 1, 3], 2: [4, 5]}, 10, 8)
        examples = []
        for index, t in [(0, 1), (0, 2), (0, 3), (1, 1)]:
            history = dataset.histories[index]
            with patch.object(
                np.random, "randint", side_effect=[t] + [10] * NEGATIVES
            ) as draw, patch.object(
                np.random, "random", side_effect=lambda size: np.ones(size)
            ):
                sequence, candidates = dataset[index]
            assert draw.call_args_list[0].args == (1, len(history))
            assert np.array_equal(sequence[-t:], history[:t])
            assert not sequence[:-t].any()
            assert candidates[0] == history[t]
            assert all(item not in dataset.known[index] for item in candidates[1:])
            assert sequence.shape == (8,)
            assert candidates.shape == (1 + NEGATIVES,)
            examples.append((sequence, candidates))
        # At t=2, the earlier occurrence of target=1 must remain in context.
        assert examples[1][0][-2] == examples[1][1][0] == 1
        # Dropping every context item must trigger the nonempty safeguard.
        with patch.object(
            np.random, "randint", side_effect=[3, 0] + [10] * NEGATIVES
        ), patch.object(
            np.random, "random", side_effect=lambda size: np.zeros(size)
        ):
            sequence, candidates = dataset[0]
        assert np.count_nonzero(sequence) == 1
        assert sequence[-1] == dataset.histories[0][0]
        assert np.stack([x[0] for x in examples]).shape == (4, 8)
        assert np.stack([x[1] for x in examples]).shape == (4, 1 + NEGATIVES)
    finally:
        np.random.set_state(state)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CausalContextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads=2, dropout=0.1, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * dim, dim),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, context, attention_mask, mask):
        normalized = self.attention_norm(context)
        attended = self.attention(
            normalized, normalized, normalized,
            attn_mask=attention_mask, need_weights=False,
        )[0]
        context = context + self.dropout(attended)
        context = context.masked_fill(~mask.unsqueeze(-1), 0.0)
        context = context + self.dropout(
            self.feedforward(self.feedforward_norm(context))
        )
        return context.masked_fill(~mask.unsqueeze(-1), 0.0)


class AttentiveSimilarity(nn.Module):
    def __init__(self, itemnum, dim=96, maxlen=128, attention_dim=32):
        super().__init__()
        self.itemnum = itemnum
        self.dim = dim
        self.maxlen = maxlen
        self.context = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.target = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.history_key = nn.Linear(dim, attention_dim, bias=False)
        self.candidate_key = nn.Linear(dim, attention_dim, bias=False)
        self.attention_dim = attention_dim
        self.recency_bias = nn.Parameter(torch.zeros(8))
        self.dropout = nn.Dropout(0.1)
        nn.init.normal_(self.context.weight, std=0.08)
        nn.init.normal_(self.target.weight, std=0.08)
        nn.init.zeros_(self.bias.weight)
        with torch.no_grad():
            self.context.weight[0].zero_()
            self.target.weight[0].zero_()
        # Preserve direct similarity initially; learn the contextual contribution.
        self.position = nn.Embedding(maxlen, dim)
        self.encoder_blocks = nn.ModuleList(
            [CausalContextBlock(dim) for _ in range(2)]
        )
        self.context_gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.position.weight, std=0.02)

    def encode_context(self, history, original_context):
        mask = history.ne(0)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        context = original_context + self.position(positions)
        context = context.masked_fill(~mask.unsqueeze(-1), 0.0)
        length = history.size(1)
        future = torch.ones(
            length, length, dtype=torch.bool, device=history.device
        ).triu(1)
        # Real queries exclude future positions and padding keys. Padded
        # queries attend only themselves, including for entirely empty rows.
        blocked = future.unsqueeze(0) | ~mask.unsqueeze(1)
        self_only = ~torch.eye(length, dtype=torch.bool, device=history.device)
        blocked = torch.where(mask.unsqueeze(-1), blocked, self_only.unsqueeze(0))
        attention_mask = blocked.repeat_interleave(2, dim=0)
        for block in self.encoder_blocks:
            context = block(context, attention_mask, mask)
        return context

    def forward(self, history, candidates):
        mask = history.ne(0)
        r = (mask.long().flip(-1).cumsum(-1).flip(-1) - 1).clamp_min(0)
        r = r.masked_fill(~mask, 0)
        bucket = torch.floor(torch.log2(1.0 + r.float())).long().clamp(max=7)
        original_context = self.dropout(self.context(history))
        encoded_context = self.encode_context(history, original_context)
        context = original_context + torch.tanh(self.context_gate) * encoded_context
        target = self.target(candidates)
        keys = torch.tanh(self.history_key(context))
        queries = torch.tanh(self.candidate_key(target))
        compatibility = torch.bmm(queries, keys.transpose(1, 2))
        compatibility = compatibility / math.sqrt(self.attention_dim)
        # Bounded attention and sublinear normalization follow NAIS's
        # motivation: ordinary softmax over-penalizes long user histories.
        # Zero initialization preserves parent scores; signed, bounded bucket
        # biases learn recency within the existing retained context.
        attention = torch.exp(
            3.0 * torch.tanh(compatibility)
            + 2.0 * torch.tanh(self.recency_bias[bucket]).unsqueeze(1)
        )
        attention = attention * mask.unsqueeze(1)
        pair_scores = torch.bmm(target, context.transpose(1, 2))
        normalizer = attention.sum(-1).clamp_min(1.0).pow(0.5)
        scores = (attention * pair_scores).sum(-1) / normalizer
        return scores + self.bias(candidates).squeeze(-1)

    def predict(self, history, candidates):
        return self(history, candidates)


def check_attentive_similarity():
    import io

    # These checks use CPU only and restore its RNG without seeding CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        config = dict(itemnum=8, dim=96, maxlen=12, attention_dim=32)
        model = AttentiveSimilarity(**config).eval()
        history = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 2, 1, 3],
            [1, 2, 3, 4, 5, 6],
        ])
        candidates = torch.tensor([[2, 3, 4]]).expand(3, -1)
        mask = history.ne(0)

        def parent_scores():
            context = model.context(history)
            target = model.target(candidates)
            keys = torch.tanh(model.history_key(context))
            queries = torch.tanh(model.candidate_key(target))
            compatibility = torch.bmm(queries, keys.transpose(1, 2))
            compatibility = compatibility / math.sqrt(model.attention_dim)
            distance = (mask.long().flip(-1).cumsum(-1).flip(-1) - 1).clamp_min(0)
            bucket = torch.floor(torch.log2(1.0 + distance.float())).long().clamp(max=7)
            attention = torch.exp(
                3.0 * torch.tanh(compatibility)
                + 2.0 * torch.tanh(model.recency_bias[bucket]).unsqueeze(1)
            ) * mask.unsqueeze(1)
            pair_scores = torch.bmm(target, context.transpose(1, 2))
            normalizer = attention.sum(-1).clamp_min(1.0).pow(0.5)
            return (
                (attention * pair_scores).sum(-1) / normalizer
                + model.bias(candidates).squeeze(-1)
            )

        with torch.no_grad():
            assert model.context_gate.item() == 0.0
            model.recency_bias.copy_(torch.linspace(-0.7, 0.7, 8))
            torch.testing.assert_close(
                model.predict(history, candidates), parent_scores(), rtol=0, atol=0
            )
            model.context_gate.fill_(0.5)
            model.bias.weight[2:].fill_(0.1)
            encoded = model.encode_context(history, model.context(history))
            assert torch.count_nonzero(encoded[~mask]).item() == 0
            scores = model.predict(history, candidates)
            assert torch.isfinite(scores).all()
            torch.testing.assert_close(
                scores[0], model.bias(candidates[0]).squeeze(-1), rtol=0, atol=0
            )
            # Even corrupt padding embeddings cannot affect real queries/scores.
            model.context.weight[0].fill_(100.0)
            torch.testing.assert_close(model.predict(history, candidates), scores)
            compact = torch.tensor([[0, 0, 1, 2, 1, 3]])
            torch.testing.assert_close(
                model.predict(compact, candidates[1:2]), scores[1:2]
            )
            # Enabling the gate must not let a changed suffix affect its prefix.
            changed = history.clone()
            changed[:, 4:] = torch.tensor([7, 8])
            changed_encoded = model.encode_context(changed, model.context(changed))
            torch.testing.assert_close(encoded[:, :4], changed_encoded[:, :4])
            short_encoded = model.encode_context(
                history[:, :4], model.context(history[:, :4])
            )
            torch.testing.assert_close(encoded[:, :4], short_encoded)
            mixed = model.context(history) + model.context_gate.tanh() * encoded
            changed_mixed = (
                model.context(changed) + model.context_gate.tanh() * changed_encoded
            )
            torch.testing.assert_close(mixed[mask & (torch.arange(6) < 4)],
                                       changed_mixed[mask & (torch.arange(6) < 4)])
            # Empty rows also need finite training gradients, not only scores.
        model.zero_grad(set_to_none=True)
        model(history, candidates).sum().backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        with torch.no_grad():
            scores = model.predict(history, candidates)
            buffer = io.BytesIO()
            torch.save({'config': config, 'state_dict': model.state_dict()}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location='cpu', weights_only=True)
            restored = AttentiveSimilarity(**checkpoint['config']).eval()
            restored.load_state_dict(checkpoint['state_dict'], strict=True)
            torch.testing.assert_close(
                restored.predict(history, candidates), scores, rtol=0, atol=0
            )


def make_validation(train, valid, itemnum, maxlen):
    # Dedicated RNG: training does not change the evaluation population.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        history = train[user][-maxlen:]
        seq = np.zeros(maxlen, dtype=np.int64)
        if history:
            seq[-len(history) :] = history
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Validation requires 99 unique unseen negatives")
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
        torch.from_numpy(np.stack(sequences)).long(),
        torch.tensor(candidates, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate(model, evaluation, device):
    model.eval()
    sequences, candidates = evaluation
    hits = 0.0
    ndcg = 0.0
    for start in range(0, len(sequences), 32):
        scores = model.predict(
            sequences[start : start + 32].to(device),
            candidates[start : start + 32].to(device),
        )
        # Stable candidate-order tie breaking, consistent with positive first.
        ranks = (scores[:, 1:] > scores[:, :1]).sum(1).cpu().numpy()
        hits += float((ranks < 10).sum())
        ndcg += float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0).sum())
    return hits / len(sequences), ndcg / len(sequences)


def train_category(name, device):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    dataset = PurchaseDataset(train, itemnum, MAXLEN)
    if not len(dataset):
        raise ValueError("No eligible training histories for " + name)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    config = dict(itemnum=itemnum, dim=DIM, maxlen=MAXLEN, attention_dim=32)
    model = AttentiveSimilarity(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=1e-5, betas=(0.9, 0.98)
    )
    evaluation = make_validation(train, valid, itemnum, MAXLEN)
    criterion = nn.CrossEntropyLoss()
    best = (-1.0, -1.0)
    best_epoch = 0
    best_state = None
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for histories, candidates in loader:
            histories = histories.to(device, non_blocking=True)
            candidates = candidates.to(device, non_blocking=True)
            scores = model(histories, candidates)
            labels = torch.zeros(len(histories), dtype=torch.long, device=device)
            loss = criterion(scores, labels)
            # Penalize the embeddings actually used by this batch.
            reg = (
                model.context(histories).square().sum(-1) * histories.ne(0)
            ).sum() / histories.ne(0).sum().clamp_min(1)
            reg = reg + model.target(candidates).square().sum(-1).mean()
            loss = loss + 1e-4 * reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(histories)
            examples += len(histories)
        if epoch == 1 or epoch % 2 == 0 or epoch == EPOCHS:
            metric = evaluate(model, evaluation, device)
            if metric > best:
                best = metric
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(
                f"{name} epoch={epoch} loss={total_loss / examples:.5f} "
                f"validation_HR@10={metric[0]:.5f} "
                f"validation_NDCG@10={metric[1]:.5f}",
                flush=True,
            )
        else:
            print(f"{name} epoch={epoch} loss={total_loss / examples:.5f}", flush=True)
        if epoch - best_epoch >= PATIENCE:
            break
        if time.monotonic() - started >= SECONDS_PER_CATEGORY:
            break
    model.load_state_dict(best_state, strict=True)
    hr, _ = evaluate(model, evaluation, device)
    torch.save(
        {"config": config, "state_dict": best_state},
        os.path.join(WORKING, name + "_model.pt"),
    )
    check_predictor(name, model, evaluation)
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return hr


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific',
          'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(name):
    global _CLASS
    if name not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(name))
    if name in _MODELS:
        return _MODELS[name]
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'attentive_similarity_train', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.AttentiveSimilarity
    checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'),
                            map_location='cpu', weights_only=True)
    # Strict loading below verifies every encoder parameter as well as these
    # architectural markers. Reconstruct dimensions from the saved config.
    required = ('recency_bias', 'context_gate', 'position.weight',
                'encoder_blocks.0.attention.in_proj_weight',
                'encoder_blocks.1.attention.in_proj_weight')
    missing = [key for key in required if key not in checkpoint['state_dict']]
    if missing:
        raise ValueError('Incomplete contextual checkpoint: ' + str(missing))
    model = _CLASS(**checkpoint['config']).to(_DEVICE)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    _MODELS[name] = model
    return model


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    name = dataset_name if dataset_name is not None else _NAMES[0]
    model = _load(name)
    if not candidates:
        return []
    hist = list(history)[-model.maxlen:]
    seq = [0] * (model.maxlen - len(hist)) + hist
    sequence = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    outputs = []
    for start in range(0, len(candidates), 256):
        candidate_tensor = torch.tensor([candidates[start:start + 256]],
                                        dtype=torch.long, device=_DEVICE)
        outputs.extend(model.predict(sequence, candidate_tensor)[0].cpu().tolist())
    return outputs
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(script)


@torch.inference_mode()
def check_predictor(name, model, evaluation):
    # Exercise the emitted predictor and its actual checkpoint loader. Preserve
    # RNG state because constructing the restored model initializes parameters.
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        spec = importlib.util.spec_from_file_location(
            'attentive_similarity_predict_check', os.path.join(WORKING, 'predict.py')
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model.eval()
        sequences, candidates = evaluation
        device = next(model.parameters()).device
        for index in range(min(3, len(sequences))):
            sequence = sequences[index:index + 1].to(device)
            # Also exercise the predictor's 256-candidate chunk boundary.
            choices = candidates[index].repeat(3) if index == 0 else candidates[index]
            # Match the emitted predictor's single-user candidate chunks exactly.
            expected = torch.cat([
                model.predict(
                    sequence,
                    choices[start:start + 256].unsqueeze(0).to(device),
                )[0].cpu()
                for start in range(0, len(choices), 256)
            ])
            history = sequences[index][sequences[index].ne(0)].tolist()
            actual = torch.tensor(module.predict(
                index, history, choices.tolist(), dataset_name=name
            ), dtype=expected.dtype)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        module._MODELS.clear()


def main():
    seed_everything(42)
    check_purchase_dataset()
    check_attentive_similarity()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_predict_script()
    metrics = [train_category(name, device) for name in DATASETS]
    for name, hr in zip(DATASETS, metrics):
        print(f"{name} selected_validation_HR@10={hr:.5f}", flush=True)
    score = float(np.mean(metrics))
    print(
        f"Equal-weight macro validation HR@10={score:.5f}; "
        f"delta_vs_parent_0.4313={score - 0.4313:+.5f}; "
        f"delta_vs_SASRec_0.505={score - 0.505:+.5f}",
        flush=True,
    )
    print(f"Final Validation Score: {score}", flush=True)


if __name__ == "__main__":
    import traceback

    os.makedirs(WORKING, exist_ok=True)
    try:
        main()
    except Exception:
        details = traceback.format_exc()
        with open(os.path.join(WORKING, "training_error.log"), "w") as f:
            f.write(details)
        raise
