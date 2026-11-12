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
NEGATIVES = 64
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
        # Supervise a purchase using only its preceding interactions.
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


def check_purchase_sampler():
    from unittest.mock import patch

    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        original_randint = np.random.randint

        def check(history, maxlen, t, drop_all=False, recent=True):
            dataset = PurchaseDataset({0: history}, 23, maxlen)
            first_draw = True

            def randint(low, high=None, size=None, dtype=int):
                nonlocal first_draw
                if first_draw:
                    first_draw = False
                    assert (low, high, size) == (1, len(history), None)
                    return t
                return original_randint(low, high, size=size, dtype=dtype)

            def random_draw(size=None):
                if size is None:
                    return 0.0 if recent else 0.9
                return np.zeros(size) if drop_all else np.ones(size)

            with patch.object(np.random, "randint", side_effect=randint), patch.object(
                np.random, "random", side_effect=random_draw
            ):
                sequence, candidates = dataset[0]
            assert not first_draw
            assert sequence.shape == (maxlen,)
            assert candidates.shape == (NEGATIVES + 1,)
            assert sequence.dtype == candidates.dtype == np.dtype(np.int64)
            assert candidates[0] == history[t]
            assert np.all((candidates[1:] >= 1) & (candidates[1:] <= 23))
            assert all(int(item) not in dataset.known[0] for item in candidates[1:])
            context = sequence[sequence != 0].tolist()
            assert context
            assert not sequence[:maxlen - len(context)].any()
            prefix = iter(history[:t])
            assert all(any(past == item for past in prefix) for item in context)
            if not drop_all and t <= maxlen:
                assert context == history[:t]
            if not drop_all and t > maxlen:
                assert len(context) == maxlen
                if recent:
                    assert context == history[t - maxlen:t]
            return sequence, candidates

        # Both target boundaries, including the shortest eligible history.
        check([1, 2], 8, 1)
        check([1, 2, 3, 4], 8, 1)
        check([1, 2, 3, 4], 8, 3)
        # An earlier occurrence of the target remains in its prefix.
        sequence, _ = check([1, 2, 1, 3], 8, 2)
        assert sequence[-2:].tolist() == [1, 2]
        check([1, 1, 2], 8, 1)
        # Force all context dropout draws to fail; the safeguard keeps one.
        sequence, _ = check([1, 2, 3, 4], 8, 3, drop_all=True)
        assert np.count_nonzero(sequence) == 1
        # Exercise both truncation branches and ordered subsampling.
        check(list(range(1, 13)), 4, 11, recent=True)
        check(list(range(1, 13)), 4, 11, recent=False)
        # Only item 23 is unseen, including purchases after the target.
        _, candidates = check(list(range(1, 23)), 8, 1)
        assert np.all(candidates[1:] == 23)
    finally:
        np.random.set_state(rng_state)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AttentiveSimilarity(nn.Module):
    def __init__(self, itemnum, dim=96, maxlen=128, attention_dim=32,
                 encoder_dropout=0.1):
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
        self.dropout = nn.Dropout(0.1)
        # One causal block supplies order-aware context; candidate scoring stays
        # unchanged. The initially conservative gate favors the original context.
        self.position = nn.Embedding(maxlen, dim)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=1, dim_feedforward=2 * dim,
            dropout=encoder_dropout, batch_first=True,
        )
        self.encoder_projection = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.normal_(self.position.weight, std=0.08)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.normal_(self.context.weight, std=0.08)
        nn.init.normal_(self.target.weight, std=0.08)
        nn.init.zeros_(self.bias.weight)
        with torch.no_grad():
            self.context.weight[0].zero_()
            self.target.weight[0].zero_()

    def _encode_context(self, context, mask):
        # Compact nonempty rows in purchase order. Every attention query can
        # attend to at least its first real token, including padded queries.
        # Empty histories bypass attention entirely to avoid all-masked rows.
        encoded = torch.zeros_like(context)
        rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return encoded
        selected_mask = mask[rows]
        lengths = selected_mask.sum(dim=1)
        width = int(lengths.max().item())
        row_ids, original_positions = selected_mask.nonzero(as_tuple=True)
        compact_positions = selected_mask.long().cumsum(dim=1)[
            row_ids, original_positions
        ] - 1
        compact = context.new_zeros((len(rows), width, self.dim))
        compact[row_ids, compact_positions] = context[
            rows[row_ids], original_positions
        ]
        positions = torch.arange(width, device=context.device)
        compact_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        compact = (compact + self.position(positions).unsqueeze(0))
        compact = compact * compact_mask.unsqueeze(-1)
        causal_mask = torch.ones(
            width, width, dtype=torch.bool, device=context.device
        ).triu(diagonal=1)
        states = self.encoder(
            compact, src_mask=causal_mask,
            src_key_padding_mask=~compact_mask,
        )
        states = states.masked_fill(~compact_mask.unsqueeze(-1), 0.0)
        states = self.encoder_projection(states)
        states = torch.nn.functional.normalize(states, dim=-1, eps=1e-8)
        scale = context[rows[row_ids], original_positions].norm(
            dim=-1, keepdim=True
        )
        encoded[rows[row_ids], original_positions] = (
            states[row_ids, compact_positions] * scale
        )
        return encoded.masked_fill(~mask.unsqueeze(-1), 0.0)

    def forward(self, history, candidates):
        mask = history.ne(0)
        context = self.dropout(self.context(history))
        encoded = self._encode_context(context, mask)
        g = torch.sigmoid(self.gate(torch.cat((context, encoded), dim=-1)))
        fused = ((1 - g) * context + g * encoded) * mask.unsqueeze(-1)
        target = self.target(candidates)
        keys = torch.tanh(self.history_key(fused))
        queries = torch.tanh(self.candidate_key(target))
        compatibility = torch.bmm(queries, keys.transpose(1, 2))
        compatibility = compatibility / math.sqrt(self.attention_dim)
        # Bounded attention and sublinear normalization follow NAIS's
        # motivation: ordinary softmax over-penalizes long user histories.
        attention = torch.exp(3.0 * torch.tanh(compatibility))
        attention = attention * mask.unsqueeze(1)
        pair_scores = torch.bmm(target, fused.transpose(1, 2))
        normalizer = attention.sum(-1).clamp_min(1.0).pow(0.5)
        scores = (attention * pair_scores).sum(-1) / normalizer
        return scores + self.bias(candidates).squeeze(-1)

    def predict(self, history, candidates):
        return self(history, candidates)


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
    config = dict(
        itemnum=itemnum, dim=DIM, maxlen=MAXLEN, attention_dim=32,
        encoder_dropout=0.1,
    )
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
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reference = {
        "Movies_and_TV": 0.425,
        "Industrial_and_Scientific": 0.495,
        "Electronics": 0.365,
        "CDs_and_Vinyl": 0.480,
    }
    sasrec = {
        "Movies_and_TV": 0.580,
        "Industrial_and_Scientific": 0.540,
        "Electronics": 0.420,
        "CDs_and_Vinyl": 0.480,
    }
    print(
        f"{name} selected_validation_HR@10={hr:.5f} "
        f"delta_reference={hr - reference[name]:+.5f} "
        f"delta_SASRec={hr - sasrec[name]:+.5f}",
        flush=True,
    )
    return hr


def check_causal_encoder():
    # Tiny CPU checks leave training RNG state unchanged and use no split data.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = AttentiveSimilarity(12, dim=8, maxlen=6, attention_dim=4)
        model.eval()
        history = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 2, 3],
        ])
        candidates = torch.tensor([[4, 5, 6]]).expand(3, -1)
        mask = history.ne(0)
        encoded = model._encode_context(model.context(history), mask)
        assert torch.isfinite(encoded).all()
        assert torch.count_nonzero(encoded[~mask]).item() == 0
        scores = model(history, candidates)
        assert torch.isfinite(scores).all()
        torch.testing.assert_close(
            scores[0], model.bias(candidates[0]).squeeze(-1)
        )
        changed = history.clone()
        changed[2, -1] = 7
        changed_encoded = model._encode_context(
            model.context(changed), changed.ne(0)
        )
        # With dropout disabled, later purchases cannot change earlier states.
        torch.testing.assert_close(
            encoded[2, 3:5], changed_encoded[2, 3:5],
            rtol=1e-5, atol=1e-6,
        )
        loss = nn.CrossEntropyLoss()(scores, torch.zeros(3, dtype=torch.long))
        loss.backward()
        for parameter in (
            model.position.weight,
            model.encoder.self_attn.in_proj_weight,
            model.encoder.linear1.weight,
            model.encoder_projection.weight,
            model.gate.weight,
            model.gate.bias,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum().item() > 0


@torch.inference_mode()
def check_generated_predictor(device):
    spec = importlib.util.spec_from_file_location(
        "gated_similarity_predict_check", os.path.join(WORKING, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    for name in DATASETS:
        checkpoint = torch.load(
            os.path.join(WORKING, name + "_model.pt"),
            map_location="cpu", weights_only=True,
        )
        model = AttentiveSimilarity(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        # Include empty input, left padding, truncation, and chunk boundaries.
        candidates = [1 + i % model.itemnum for i in range(259)]
        for history in (
            [], [1],
            [1 + i % model.itemnum for i in range(model.maxlen + 3)],
        ):
            hist = history[-model.maxlen:]
            seq = [0] * (model.maxlen - len(hist)) + hist
            expected = model.predict(
                torch.tensor([seq], dtype=torch.long, device=device),
                torch.tensor([candidates], dtype=torch.long, device=device),
            )[0].cpu()
            actual = torch.tensor(predictor.predict(
                0, history, candidates, dataset_name=name
            ))
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            assert predictor.predict(0, history, [], dataset_name=name) == []
        del model, checkpoint
        predictor._MODELS.clear()
    print("Causal encoder and checkpoint/predictor checks passed", flush=True)


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
    model = _CLASS(**checkpoint['config']).to(_DEVICE)
    # Saved configuration reconstructs the encoder, projection, and gate;
    # strict loading rejects any architecture/checkpoint mismatch.
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


def main():
    seed_everything(42)
    check_purchase_sampler()
    check_causal_encoder()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = [train_category(name, device) for name in DATASETS]
    write_predict_script()
    check_generated_predictor(device)
    score = float(np.mean(metrics))
    print(f"Final Validation Score: {score}")
    print(
        f"Macro validation HR@10={score:.5f} "
        f"reference=0.44120 delta={score - 0.4412:+.5f} "
        f"SASRec=0.50500 delta={score - 0.505:+.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
