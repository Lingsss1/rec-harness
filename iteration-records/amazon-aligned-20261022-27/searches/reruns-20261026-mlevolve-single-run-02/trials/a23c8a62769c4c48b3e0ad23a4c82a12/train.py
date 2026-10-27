from __future__ import annotations

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
from torch.nn import functional as F
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
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "64"))
NEGATIVES = 256
# Four sequential categories fit inside the nine-hour execution limit.
CATEGORY_SECONDS = float(os.environ.get("SASREC_CATEGORY_SECONDS", "6500"))


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
            values = line.split()
            if len(values) == 2:
                result[int(values[0])].append(int(values[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    path = os.path.join(DATA_DIR, "dataset_stats.json")
    itemnum = max(max(items) for items in train.values() if items)
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if name in stats:
            itemnum = int(stats[name]["itemnum"])
    # Existing prepared splits are consumed unchanged; no preprocessing is refit.
    return train, valid, itemnum


def pack_histories(histories, maxlen):
    result = np.zeros((len(histories), maxlen), dtype=np.int64)
    for row, history in enumerate(histories):
        tail = history[-maxlen:]
        if tail:
            result[row, -len(tail) :] = tail
    return result


class SequenceDataset(Dataset):
    def __init__(self, train, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No eligible training histories")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        h = self.histories[index]
        end = len(h)
        # Prefix augmentation uses only the supplied training interactions.
        if len(h) > 2 and random.random() < 0.25:
            end = random.randint(2, len(h))
        window = h[max(0, end - self.maxlen - 1) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        n = len(window) - 1
        seq[-n:] = window[:-1]
        pos[-n:] = window[1:]
        return seq, pos, index


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class CausalBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.dropout = nn.Dropout(dropout)
        self.rate = dropout

    def forward(self, x, allowed, valid):
        b, length, hidden = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.rate if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(b, length, hidden)
        x = x + self.dropout(self.out(y))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x * valid.unsqueeze(-1)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden=96, blocks=2, heads=2, dropout=0.2, maxlen=64):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.hidden = hidden
        self.item_emb = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [CausalBlock(hidden, heads, dropout) for _ in range(blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        # One jointly trained history branch shares the existing item embeddings.
        self.history_query = nn.Parameter(torch.empty(hidden))
        self.history_proj = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden * 2, hidden)
        nn.init.normal_(self.history_query, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def _history_pool(self, seq):
        embeddings = self.item_emb(seq)
        length = seq.shape[1]
        visible = torch.ones(
            length, length, device=seq.device, dtype=torch.bool
        ).tril()[None] & seq.ne(0)[:, None, :]
        has_history = visible.any(dim=-1, keepdim=True)
        logits = (embeddings * self.history_query).sum(-1) / math.sqrt(self.hidden)
        logits = logits[:, None, :].expand(-1, length, -1)
        logits = logits.masked_fill(~visible, float("-inf"))
        # Empty prefixes use finite softmax inputs and exactly zero weights.
        logits = torch.where(has_history, logits, torch.zeros_like(logits))
        attention = torch.softmax(logits, dim=-1).masked_fill(~visible, 0.0)
        pooled = attention @ embeddings
        pooled = self.history_proj(pooled)
        return pooled.masked_fill(~has_history, 0.0)

    def log2feats(self, seq):
        b, length = seq.shape
        valid = seq.ne(0)
        positions = torch.arange(length, device=seq.device)
        x = self.item_emb(seq) * math.sqrt(self.hidden)
        x = self.emb_dropout(x + self.pos_emb(positions)[None])
        x = x * valid.unsqueeze(-1)
        causal = torch.ones(length, length, device=seq.device, dtype=torch.bool).tril()
        diagonal = torch.eye(length, device=seq.device, dtype=torch.bool)
        # Padding queries retain a valid diagonal to avoid all-masked attention.
        allowed = ((causal[None] & valid[:, None, :]) | diagonal[None])[:, None]
        for block in self.blocks:
            x = block(x, allowed, valid)
        pooled = self._history_pool(seq)
        g = torch.sigmoid(self.gate(torch.cat([x, pooled], dim=-1)))
        x = (g * x + (1.0 - g) * pooled) * valid.unsqueeze(-1)
        return F.normalize(self.final_norm(x), dim=-1)

    def scale(self):
        return self.log_scale.clamp(math.log(1.0), math.log(40.0)).exp()

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1]
        items = F.normalize(self.item_emb(candidates), dim=-1)
        scores = (items * features[:, None]).sum(-1) * self.scale()
        return scores + self.item_bias(candidates).squeeze(-1)

    def ranking_loss(self, seq, pos, negatives, excluded):
        features = self.log2feats(seq)
        valid = pos.ne(0)
        # Keep every transition for short histories and randomly sample at most
        # 24 transitions per history for long histories to bound softmax cost.
        priority = torch.rand_like(pos, dtype=torch.float32).masked_fill(~valid, -1)
        selected = priority.topk(min(24, pos.shape[1]), dim=1).indices
        rows = torch.arange(seq.shape[0], device=seq.device)[:, None]
        targets = pos[rows, selected]
        weights = torch.ones_like(selected, dtype=torch.float32)
        keep = targets.ne(0)
        weights = weights[keep]
        f = features[rows, selected][keep]
        targets = targets[keep]
        owner = rows.expand_as(selected)[keep]
        positive = (f * F.normalize(self.item_emb(targets), dim=-1)).sum(-1)
        positive = positive * self.scale() + self.item_bias(targets).squeeze(-1)
        neg_emb = F.normalize(self.item_emb(negatives), dim=-1)
        negative = (f @ neg_emb.T) * self.scale()
        negative = negative + self.item_bias(negatives).squeeze(-1)[None]
        mask = excluded[owner] | targets[:, None].eq(negatives[None])
        # Every uniform pool entry participates unless excluded; no mining or refill.
        negative = negative.masked_fill(mask, float("-inf"))
        if getattr(self, "_check_negatives", False):
            assert negatives.numel() == min(NEGATIVES, self.itemnum)
            assert negatives.unique().numel() == negatives.numel()
            assert ((negatives >= 1) & (negatives <= self.itemnum)).all().item()
            assert torch.isneginf(negative[excluded[owner]]).all().item()
            assert torch.isneginf(
                negative[targets[:, None].eq(negatives[None])]
            ).all().item()
            assert torch.isfinite(negative[~mask]).all().item()
        logits = torch.cat([positive[:, None], negative], dim=1)
        losses = F.cross_entropy(
            logits.float(),
            torch.zeros(len(targets), dtype=torch.long, device=seq.device),
            reduction="none",
        )
        if getattr(self, "_check_negatives", False):
            assert torch.isfinite(losses).all().item()
            # A positive-only denominator must yield finite, exactly zero CE.
            assert losses[mask.all(dim=1)].eq(0).all().item()
        # Weight selected valid transitions uniformly within each history;
        # normalize within each history and give histories with valid selected
        # targets equal aggregate influence.
        loss_sums = torch.zeros(seq.shape[0], dtype=torch.float32, device=seq.device)
        weight_sums = torch.zeros_like(loss_sums)
        loss_sums.scatter_add_(0, owner, losses.float() * weights)
        weight_sums.scatter_add_(0, owner, weights)
        history_losses = loss_sums / weight_sums.clamp_min(1e-12)
        return history_losses.sum() / weight_sums.gt(0).sum().clamp_min(1)


def validation_arrays(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    users = [u for u in users if valid[u]]
    histories, candidates = [], []
    for user in users:
        history = train[user]
        excluded = set(history) | set(valid[user]) | {0}
        if itemnum - len(excluded - {0}) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        histories.append(history)
        candidates.append([valid[user][0]] + list(negatives))
    if not users:
        raise ValueError("Empty validation split")
    return (pack_histories(histories, maxlen), np.asarray(candidates, dtype=np.int64))


@torch.inference_mode()
def evaluate(model, arrays, device):
    model.eval()
    sequences, candidates = arrays
    hits = 0
    for start in range(0, len(sequences), 128):
        seq = torch.as_tensor(sequences[start : start + 128], device=device)
        cand = torch.as_tensor(candidates[start : start + 128], device=device)
        scores = model.predict(seq, cand)
        # Positive is candidate zero; stable sorting places it first on ties.
        rank = (scores[:, 1:] > scores[:, :1]).sum(dim=1)
        hits += int((rank < 10).sum().item())
    return hits / len(sequences)


def train_category(name, device):
    seed_all(42)
    start_time = time.monotonic()
    train, valid, itemnum = load_data(name)
    config = dict(
        itemnum=itemnum, hidden=96, blocks=2, heads=2, dropout=0.2, maxlen=MAXLEN
    )
    model = SASRec(**config).to(device)
    # Deterministic forward-only checks consume no training samples or RNG draws.
    # Include an empty history and a left-padded history with an edited future.
    with torch.no_grad():
        model.eval()
        probe = torch.zeros((2, MAXLEN), dtype=torch.long, device=device)
        width = min(MAXLEN, 6)
        probe[1, -width:] = (
            torch.arange(width, device=device) % itemnum + 1
        )
        pooled = model._history_pool(probe)
        if not torch.isfinite(pooled).all().item():
            raise FloatingPointError("Nonfinite pooled history features")
        torch.testing.assert_close(pooled[0], torch.zeros_like(pooled[0]))
        changed = probe.clone()
        changed[1, -1] = changed[1, -1] % itemnum + 1
        original_features = model.log2feats(probe)
        changed_features = model.log2feats(changed)
        if not torch.isfinite(original_features).all().item():
            raise FloatingPointError("Nonfinite gated sequence features")
        earlier_valid = probe.ne(0)
        earlier_valid[:, -1] = False
        torch.testing.assert_close(
            original_features[earlier_valid], changed_features[earlier_valid],
            rtol=1e-5, atol=1e-6,
        )
        model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    dataset = SequenceDataset(train, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=BATCH * STEPS, generator=generator
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=2,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=True,
    )
    val_arrays = validation_arrays(train, valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best, best_epoch = -1.0, 0
    neg_rng = np.random.default_rng(42)
    completed = 0
    stop = False
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, batches = 0.0, 0
        # Mild decay preserves useful learning rates throughout early stopping.
        lr = 8e-4 * (
            0.35 + 0.65 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(EPOCHS, 1)))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        for seq, pos, indices in loader:
            # One distinct, uniform nonzero candidate pool shared by this batch.
            neg_np = (
                neg_rng.choice(itemnum, size=min(NEGATIVES, itemnum), replace=False)
                .astype(np.int64, copy=False) + 1
            )
            # Membership depends exclusively on complete training histories.
            excluded = np.stack(
                [
                    np.isin(neg_np, dataset.histories[int(index)], assume_unique=False)
                    for index in indices.tolist()
                ]
            )
            model._check_negatives = completed == 0 and batches == 0
            if model._check_negatives:
                assert len(neg_np) == min(NEGATIVES, itemnum)
                assert len(np.unique(neg_np)) == len(neg_np)
                assert np.all((neg_np >= 1) & (neg_np <= itemnum))
                # Independently verify membership against complete histories,
                # including interactions outside each augmented training prefix.
                for row, index in enumerate(indices.tolist()):
                    history_items = set(dataset.histories[int(index)].tolist())
                    expected = np.fromiter(
                        (int(item) in history_items for item in neg_np),
                        dtype=np.bool_, count=len(neg_np),
                    )
                    assert np.array_equal(excluded[row], expected)
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            negatives = torch.as_tensor(neg_np, device=device)
            excluded_t = torch.as_tensor(excluded, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.ranking_loss(seq, pos, negatives, excluded_t)
            loss.backward()
            # Check the first real training backward without an extra update.
            if completed == 0 and batches == 0:
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite ranking loss")
                for parameter_name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        raise RuntimeError("Missing gradient: " + parameter_name)
                    if not torch.isfinite(parameter.grad).all().item():
                        raise FloatingPointError(
                            "Nonfinite gradient: " + parameter_name
                        )
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            if time.monotonic() - start_time > CATEGORY_SECONDS:
                stop = True
                break
        completed = epoch
        # Fixed validation candidates and users are reused for every check.
        checked = epoch == 1 or epoch % 5 == 0 or stop or epoch == EPOCHS
        if checked:
            score = evaluate(model, val_arrays, device)
            if score > best:
                best, best_epoch = score, epoch
                torch.save(
                    {
                        "config": config,
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "validation_hr10": score,
                        # Small reference verifies reconstruction of the chosen
                        # single checkpoint, including all new internal parameters.
                        "reload_probe_scores": model.predict(
                            torch.as_tensor(val_arrays[0][:2], device=device),
                            torch.as_tensor(val_arrays[1][:2], device=device),
                        ).detach().cpu(),
                    },
                    checkpoint,
                )
            print(
                f"{name} epoch={epoch} loss={total / max(batches, 1):.5f} "
                f"val_HR@10={score:.5f} best={best:.5f}",
                flush=True,
            )
        else:
            print(
                f"{name} epoch={epoch} loss={total / max(batches, 1):.5f}", flush=True
            )
        if stop or epoch - best_epoch >= 35:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    del model, optimizer
    model = SASRec(**state["config"]).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        reloaded_scores = model.predict(
            torch.as_tensor(val_arrays[0][:2], device=device),
            torch.as_tensor(val_arrays[1][:2], device=device),
        )
        torch.testing.assert_close(
            reloaded_scores, state["reload_probe_scores"], rtol=1e-5, atol=1e-6
        )
    actual_score = evaluate(model, val_arrays, device)
    previous_scores = {
        "Movies_and_TV": 0.515,
        "Industrial_and_Scientific": 0.52,
        "Electronics": 0.375,
        "CDs_and_Vinyl": 0.5,
    }
    baseline_scores = {
        "Movies_and_TV": 0.58,
        "Industrial_and_Scientific": 0.54,
        "Electronics": 0.42,
        "CDs_and_Vinyl": 0.48,
    }
    print(
        f"{name} selected validation HR@10={actual_score:.5f}; "
        f"delta vs latest measured validation {previous_scores[name]:.5f}="
        f"{actual_score - previous_scores[name]:+.5f}; "
        f"delta vs baseline {baseline_scores[name]:.5f}="
        f"{actual_score - baseline_scores[name]:+.5f}",
        flush=True,
    )
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return actual_score


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODULE = None


def _get_model(dataset_name):
    global _MODULE
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset_name: ' + str(dataset_name))
    if dataset_name not in _MODELS:
        if _MODULE is None:
            spec = importlib.util.spec_from_file_location(
                'trained_sasrec_definition', os.path.join(_HERE, '..', 'train.py'))
            _MODULE = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_MODULE)
        checkpoint = torch.load(
            os.path.join(_HERE, dataset_name + '_model.pt'),
            map_location=_DEVICE, weights_only=True)
        model = _MODULE.SASRec(**checkpoint['config']).to(_DEVICE)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        _MODELS[dataset_name] = model
    return _MODELS[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    model = _get_model(dataset_name if dataset_name is not None else _DATASETS[0])
    if len(candidates) == 0:
        return []
    seq_array = _MODULE.pack_histories([list(history)], model.maxlen)
    seq = torch.as_tensor(seq_array, dtype=torch.long, device=_DEVICE)
    cand = torch.as_tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return model.predict(seq, cand)[0].float().cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(script)


def main():
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scores = {}
    for name in DATASETS:
        scores[name] = train_category(name, device)
    # Test candidates are supplied to working/predict.py by the host evaluator.
    # No hidden test files or labels are opened by this training program.
    with open(os.path.join(WORKING, "validation.json"), "w") as f:
        json.dump(
            {
                "metric": "macro HR@10",
                "evaluation_seed": 42,
                "training_seed": 42,
                "users_per_category": 200,
                "unique_negatives": 99,
                "per_category": scores,
                "macro_hr10": sum(scores.values()) / len(DATASETS),
                "latest_measured_validation_macro_hr10": 0.4775,
                "delta_vs_latest_measured_0.4775": sum(scores.values()) / len(DATASETS) - 0.4775,
                "delta_vs_reported_reference_0.5037": sum(scores.values()) / len(DATASETS) - 0.5037,
                "per_category_delta_vs_latest_measured": {
                    name: scores[name] - reference
                    for name, reference in zip(DATASETS, [0.515, 0.52, 0.375, 0.5])
                },
                "per_category_delta_vs_baseline": {
                    name: scores[name] - reference
                    for name, reference in zip(DATASETS, [0.58, 0.54, 0.42, 0.48])
                },
                "delta_vs_baseline_0.505": sum(scores.values()) / len(DATASETS) - 0.505,
            },
            f,
            indent=2,
        )
    score = sum(scores.values()) / len(DATASETS)
    print(f"Final Validation Score: {score}")
    print(
        f"Validation macro HR@10={score:.5f}; "
        f"delta vs latest measured validation 0.4775={score - 0.4775:+.5f}; "
        f"delta vs reported reference 0.5037={score - 0.5037:+.5f}; "
        f"delta vs baseline 0.505={score - 0.505:+.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
