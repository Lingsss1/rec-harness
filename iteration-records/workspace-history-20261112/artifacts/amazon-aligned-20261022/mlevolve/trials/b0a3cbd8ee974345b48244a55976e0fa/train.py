from __future__ import annotations

import importlib.util
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
from torch.utils.data import DataLoader, Dataset

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
HIDDEN = int(os.environ.get("SASREC_HIDDEN", "64"))
BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
HEADS = int(os.environ.get("SASREC_HEADS", "2"))
DROPOUT = float(os.environ.get("SASREC_DROPOUT", "0.2"))
LR = float(os.environ.get("SASREC_LR", "0.001"))
WD = float(os.environ.get("SASREC_WD", "0.00001"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))
NEGATIVES = 144


def seed_all(seed):
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


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats[name]["itemnum"])
    else:
        itemnum = max(max(items) for items in train.values() if items)
    return train, valid, itemnum


class Windows(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.seen = [set(h.tolist()) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No eligible training histories")
        if any(len(s) >= itemnum for s in self.seen):
            raise ValueError("Cannot sample unseen training negatives")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        h = self.histories[index]
        end = len(h)
        if len(h) > self.maxlen + 1 and np.random.random() < 0.3:
            end = np.random.randint(2, len(h) + 1)
        window = h[max(0, end - self.maxlen - 1) : end]
        n = len(window) - 1
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        seq[-n:] = window[:-1]
        pos[-n:] = window[1:]
        seen = self.seen[index]
        negative = np.random.randint(
            1, self.itemnum + 1, size=NEGATIVES, dtype=np.int64
        )
        for j in range(NEGATIVES):
            while int(negative[j]) in seen:
                negative[j] = np.random.randint(1, self.itemnum + 1)
        return seq, pos, negative


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class Block(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep):
        b, length, hidden = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            b, length, 3, self.heads, hidden // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attention = attention.transpose(1, 2).reshape(b, length, hidden)
        x = (x + self.drop(self.proj(attention))) * keep
        x = (x + self.drop(self.ff(self.norm2(x)))) * keep
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum,
        hidden_units=64,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        maxlen=128,
        preference_gate_bias=-2.0,
        preference_half_life=10.0,
        untied_output=True,
    ):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError("hidden_units must be divisible by num_heads")
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        if preference_half_life != 10.0:
            raise ValueError("This experiment fixes preference_half_life at 10.0")
        self.preference_half_life = float(preference_half_life)
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.drop = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [Block(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_units)
        self.preference_proj = nn.Linear(hidden_units, hidden_units)
        self.preference_norm = nn.LayerNorm(hidden_units)
        self.gate = nn.Linear(2 * hidden_units, hidden_units)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, preference_gate_bias)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        # Separate history encoding from candidate discrimination while keeping
        # existing initialization and subsequent training RNG draws unchanged.
        self.untied_output = bool(untied_output)
        if self.untied_output:
            with torch.random.fork_rng(devices=[]):
                self.output_emb = nn.Embedding(
                    itemnum + 1, hidden_units, padding_idx=0
                )
            with torch.no_grad():
                self.output_emb.weight.copy_(self.item_emb.weight)
                self.output_emb.weight[0].zero_()
        else:
            self.output_emb = self.item_emb

    def _prefix_mean(self, raw, keep):
        # Decay once per observed interaction; padding leaves both states intact.
        # Float32 recurrence preserves gradients and stability under autocast.
        decay = 2.0 ** (-1.0 / self.preference_half_life)
        sums = torch.zeros_like(raw[:, 0], dtype=torch.float32)
        counts = torch.zeros_like(keep[:, 0], dtype=torch.float32)
        outputs = []
        for t in range(raw.shape[1]):
            observed = keep[:, t].bool()
            sums = torch.where(observed, decay * sums + raw[:, t].float(), sums)
            counts = torch.where(observed, decay * counts + 1.0, counts)
            outputs.append(observed * sums / counts.clamp_min(1.0))
        return torch.stack(outputs, dim=1)

    def forward(self, seq):
        keep = seq.ne(0).unsqueeze(-1)
        raw = self.item_emb(seq)
        preference = self.preference_norm(
            self.preference_proj(self._prefix_mean(raw, keep))
        )
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = (
            self.drop(
                raw * math.sqrt(self.hidden_units)
                + self.pos_emb(positions)
            )
            * keep
        )
        for block in self.blocks:
            x = block(x, keep)
        x = self.norm(x) * keep
        g = torch.sigmoid(self.gate(torch.cat([x, preference], dim=-1)))
        return ((1 - g) * x + g * preference) * keep

    def ranking_loss(self, seq, pos, negative):
        # The sampler supplies independent draws with replacement: 128 mining
        # candidates followed by 16 uniform competitors, shared across positions.
        features = self(seq)
        valid = pos.ne(0)
        if not valid.any():
            return features.float().sum() * 0.0
        rows, columns = valid.nonzero(as_tuple=True)
        losses = []
        # Bound position-by-candidate-by-hidden allocations during both mining
        # and differentiable rescoring. Each position chooses its own hard IDs.
        for start in range(0, rows.numel(), 256):
            row = rows[start:start + 256]
            column = columns[start:start + 256]
            active_features = features[row, column]
            with torch.no_grad():
                mining_ids = negative[row, :128]
                mining_logits = torch.bmm(
                    self.output_emb(mining_ids),
                    active_features.detach().unsqueeze(-1),
                ).squeeze(-1)
                mining_logits = mining_logits + self.item_bias(
                    mining_ids
                ).squeeze(-1)
                selected = mining_logits.topk(16, dim=-1).indices
                hard_ids = mining_ids.gather(1, selected)
                negative_ids = torch.cat([hard_ids, negative[row, 128:]], dim=1)
            # Recompute selected embeddings and biases with gradients enabled.
            positive_ids = pos[row, column]
            positive_logits = (
                active_features * self.output_emb(positive_ids)
            ).sum(-1) + self.item_bias(positive_ids).squeeze(-1)
            negative_logits = torch.bmm(
                self.output_emb(negative_ids), active_features.unsqueeze(-1)
            ).squeeze(-1)
            negative_logits = negative_logits + self.item_bias(
                negative_ids
            ).squeeze(-1)
            logits = torch.cat(
                [positive_logits.unsqueeze(-1), negative_logits], dim=-1
            ).float()
            losses.append(F.cross_entropy(
                logits,
                torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device),
                reduction='none',
            ))
        losses = torch.cat(losses)
        # Count subsequent valid targets, including across padding gaps.
        d = valid.flip(1).long().cumsum(1).flip(1) - valid.long()
        weights = torch.exp2(-d.float() / 20.0) * valid.float()
        # Give each nonempty sequence equal total supervision weight.
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        return (losses * weights[valid]).sum() / valid.any(1).sum().clamp_min(1)

    def predict(self, seq, candidates):
        features = self(seq)[:, -1]
        return (self.output_emb(candidates) * features.unsqueeze(1)).sum(
            -1
        ) + self.item_bias(candidates).squeeze(-1)


def pack_history(history, maxlen):
    result = np.zeros(maxlen, dtype=np.int64)
    h = history[-maxlen:]
    if h:
        result[-len(h) :] = h
    return result


def make_validation(train, valid, itemnum, maxlen):
    # Matches the supplied benchmark protocol: fixed users and unique negatives.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        if not valid[u]:
            continue
        excluded = set(train[u]) | set(valid[u])
        if itemnum - len(excluded) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negative = set()
        while len(negative) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negative.add(item)
        sequences.append(pack_history(train[u], maxlen))
        candidates.append([valid[u][0]] + list(negative))
    if not sequences:
        raise ValueError("No validation users")
    return torch.as_tensor(np.stack(sequences)), torch.tensor(
        candidates, dtype=torch.long
    )


@torch.inference_mode()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    ranks = []
    for start in range(0, len(sequences), 100):
        scores = model.predict(
            sequences[start : start + 100].to(device),
            candidates[start : start + 100].to(device),
        ).float()
        # Positive occupies index zero; stable ordering matches the baseline on ties.
        rank = (scores[:, 1:] > scores[:, :1]).sum(1)
        ranks.append(rank.cpu())
    ranks = torch.cat(ranks)
    hr = (ranks < 10).float().mean().item()
    ndcg = torch.where(ranks < 10, 1 / torch.log2(ranks.float() + 2), 0).mean().item()
    return hr, ndcg


def train_one(name, device, deadline):
    seed_all(42)
    train, valid, itemnum = load_dataset(name)
    meta = dict(
        itemnum=itemnum,
        hidden_units=HIDDEN,
        num_blocks=BLOCKS,
        num_heads=HEADS,
        dropout_rate=DROPOUT,
        maxlen=MAXLEN,
        preference_gate_bias=-2.0,
        preference_half_life=10.0,
        untied_output=True,
    )
    model = SASRec(**meta).to(device)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    loader = DataLoader(
        Windows(train, itemnum, MAXLEN),
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42),
        drop_last=False,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WD
    )
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best = (-1.0, -1.0)
    best_epoch = 0
    path = os.path.join(WORKING, name + "_model.pt")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        completed = 0
        for step in range(STEPS):
            try:
                seq, pos, neg = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                seq, pos, neg = next(iterator)
            seq, pos, neg = [x.to(device, non_blocking=True) for x in (seq, pos, neg)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                loss = model.ranking_loss(seq, pos, neg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            completed += 1
            if time.monotonic() >= deadline:
                break
        expired = time.monotonic() >= deadline
        check = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired
        if check:
            metric = evaluate(model, validation, device)
            if metric > best:
                best = metric
                best_epoch = epoch
                torch.save(model.state_dict(), path)
            print(
                f"{name} epoch={epoch} loss={loss_sum / max(completed, 1):.5f} "
                f"validation_HR@10={metric[0]:.5f} validation_NDCG@10={metric[1]:.5f}",
                flush=True,
            )
        else:
            print(
                f"{name} epoch={epoch} loss={loss_sum / max(completed, 1):.5f}",
                flush=True,
            )
        if expired or epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True), strict=True
    )
    score, _ = evaluate(model, validation, device)
    with open(os.path.join(WORKING, name + "_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    del iterator, loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


PREDICTOR = r"""from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_pack_history = None


def _ensure_loaded():
    global _device, _pack_history
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location(
        'recommender_architecture', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _pack_history = module.pack_history
    for name in _NAMES:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        if meta.get('preference_half_life') != 10.0:
            raise ValueError('Expected fixed preference_half_life=10.0: ' + name)
        if meta.get('untied_output') is not True:
            raise ValueError('Expected untied_output=True: ' + name)
        model = module.SASRec(**meta).to(_device)
        model.load_state_dict(torch.load(os.path.join(_HERE, name + '_model.pt'),
                                        map_location=_device, weights_only=True),
                              strict=True)
        model.eval()
        _models[name] = model
        _metas[name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    name = dataset_name if dataset_name is not None else _NAMES[0]
    if name not in _models:
        raise ValueError('Unknown dataset: ' + str(name))
    if not candidates:
        return []
    seq = torch.as_tensor(_pack_history(history, _metas[name]['maxlen']),
                          dtype=torch.long, device=_device).unsqueeze(0)
    cand = torch.tensor([candidates], dtype=torch.long, device=_device)
    with torch.inference_mode():
        return _models[name].predict(seq, cand)[0].float().cpu().tolist()
"""


def verify_preference_branch():
    # Isolate diagnostic randomness so training keeps its existing seed stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        # Reproduce independent uniform draws and rejection using a separate
        # RNG; restore NumPy state so diagnostics cannot change training draws.
        numpy_state = np.random.get_state()
        try:
            np.random.seed(0)
            sampler = Windows({1: [1, 2, 3, 4, 5, 6]}, 12, 2)
            sampled_seq, sampled_pos, sampled_negative = sampler[0]
            rng = np.random.RandomState(0)
            end = 6
            if rng.random_sample() < 0.3:
                end = rng.randint(2, 7)
            window = np.arange(1, 7)[max(0, end - 3):end]
            np.testing.assert_array_equal(sampled_seq, window[:-1])
            np.testing.assert_array_equal(sampled_pos, window[1:])
            expected_negative = rng.randint(1, 13, size=144, dtype=np.int64)
            for j in range(144):
                while int(expected_negative[j]) in sampler.seen[0]:
                    expected_negative[j] = rng.randint(1, 13)
            np.testing.assert_array_equal(sampled_negative, expected_negative)
            np.testing.assert_array_equal(
                sampled_negative[:128], expected_negative[:128]
            )
            np.testing.assert_array_equal(
                sampled_negative[128:], expected_negative[128:]
            )
            assert len(sampled_negative[128:]) == 16
            assert not set(sampled_negative.tolist()) & set(range(1, 7))
            assert len(set(sampled_negative.tolist())) < len(sampled_negative)
            # Earlier training items remain excluded after window truncation;
            # the independently drawn uniform tail is neither mined nor deduped.
            assert 1 not in sampled_seq and 1 in sampler.seen[0]
            singleton = Windows({1: [1, 2]}, 12, 5)
            padded_seq, padded_pos, _ = singleton[0]
            np.testing.assert_array_equal(padded_seq, [0, 0, 0, 0, 1])
            np.testing.assert_array_equal(padded_pos, [0, 0, 0, 0, 2])
        finally:
            np.random.set_state(numpy_state)

        # Opposite features require different top-16 selections from the same
        # shared mining pool. Known IDs give a reference independent of topk.
        from unittest.mock import patch
        probe_model = SASRec(
            itemnum=146, hidden_units=8, num_blocks=0, num_heads=2,
            dropout_rate=0.0, maxlen=2,
        ).eval()
        torch.testing.assert_close(
            probe_model.item_emb.weight, probe_model.output_emb.weight
        )
        assert (probe_model.item_emb.weight.data_ptr()
                != probe_model.output_emb.weight.data_ptr())
        assert torch.count_nonzero(probe_model.output_emb.weight[0]) == 0
        with torch.no_grad():
            # Deliberately disagreeing tables expose accidental context-table
            # use in mining, positive scoring, rescoring, and prediction.
            probe_model.item_emb.weight.zero_()
            probe_model.output_emb.weight.zero_()
            probe_model.output_emb.weight[1:129, 0] = torch.linspace(-1, 1, 128)
            probe_model.output_emb.weight[129:147, 0] = 0.1
            # Biases affect selection as well as the final CE logits.
            probe_model.item_bias.weight[64] = 3.0
        probe_features = torch.zeros(1, 2, 8)
        probe_features[0, :, 0] = torch.tensor([1.0, -1.0])
        probe_features.requires_grad_(True)
        probe_seq = torch.tensor([[145, 146]])
        probe_pos = torch.tensor([[145, 146]])
        probe_negative = torch.arange(1, 145).unsqueeze(0)
        expected_hard = [
            torch.cat([torch.tensor([64]), torch.arange(114, 129)]),
            torch.cat([torch.tensor([64]), torch.arange(1, 16)]),
        ]
        with torch.no_grad():
            target_losses = []
            for column in range(2):
                candidates = torch.cat([
                    probe_pos[0, column:column + 1], expected_hard[column],
                    probe_negative[0, 128:],
                ])
                reference_logits = (
                    probe_model.output_emb(candidates) * probe_features[0, column]
                ).sum(-1) + probe_model.item_bias(candidates).squeeze(-1)
                target_losses.append(F.cross_entropy(
                    reference_logits.unsqueeze(0), torch.zeros(1, dtype=torch.long)
                ))
            first_weight = 2.0 ** (-1.0 / 20.0)
            probe_expected = (
                first_weight * target_losses[0] + target_losses[1]
            ) / (first_weight + 1.0)
        with patch.object(probe_model, 'forward', return_value=probe_features) as forward:
            probe_loss = probe_model.ranking_loss(
                probe_seq, probe_pos, probe_negative
            )
            assert forward.call_count == 1
            prediction_candidates = torch.tensor([[1, 64, 128, 145]])
            expected_scores = (
                probe_model.output_emb(prediction_candidates)
                * probe_features[:, -1].unsqueeze(1)
            ).sum(-1) + probe_model.item_bias(prediction_candidates).squeeze(-1)
            torch.testing.assert_close(
                probe_model.predict(probe_seq, prediction_candidates),
                expected_scores,
            )
        torch.testing.assert_close(probe_loss, probe_expected)
        assert torch.isfinite(probe_loss)
        probe_loss.backward()
        assert torch.isfinite(probe_features.grad).all()
        assert probe_features.grad.abs().sum() > 0
        selected_ids = torch.cat(expected_hard).unique()
        assert probe_model.item_emb.weight.grad is None
        for parameter in (probe_model.output_emb.weight, probe_model.item_bias.weight):
            assert torch.isfinite(parameter.grad).all()
            assert (parameter.grad[selected_ids].abs().sum(-1) > 0).all()
            assert (parameter.grad[129:145].abs().sum(-1) > 0).all()
            assert torch.count_nonzero(parameter.grad[16:64]) == 0
            assert torch.count_nonzero(parameter.grad[65:114]) == 0

        model = SASRec(
            itemnum=12, hidden_units=8, num_blocks=2, num_heads=2,
            dropout_rate=0.0, maxlen=5,
        ).eval()
        seq = torch.tensor([
            [0, 0, 1, 2, 3],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4],
            [1, 0, 2, 0, 1],
        ])
        torch.testing.assert_close(model.item_emb.weight, model.output_emb.weight)
        assert model.item_emb.weight.data_ptr() != model.output_emb.weight.data_ptr()
        assert torch.count_nonzero(model.output_emb.weight[0]) == 0
        with torch.no_grad():
            model.output_emb.weight.mul_(1.5)
        keep = seq.ne(0).unsqueeze(-1)
        # A nonzero padding embedding must still contribute nothing to pooling.
        with torch.no_grad():
            model.item_emb.weight[0].fill_(7.0)
            raw = model.item_emb(seq)
            means = model._prefix_mean(raw, keep)
            expected = torch.zeros_like(raw)
            # Independent reference: normalized exponential weights over observed
            # items only, including repeated IDs and gaps inside a history.
            decay = 2.0 ** (-1.0 / 10.0)
            for row in range(seq.shape[0]):
                for t in range(seq.shape[1]):
                    if not keep[row, t, 0]:
                        continue
                    observed = raw[row, :t + 1][keep[row, :t + 1, 0]].float()
                    ages = torch.arange(observed.shape[0] - 1, -1, -1).float()
                    weights = decay ** ages
                    expected[row, t] = (
                        observed * weights.unsqueeze(-1)
                    ).sum(0) / weights.sum()
            torch.testing.assert_close(means, expected)
            assert means.dtype == torch.float32
            assert torch.isfinite(means).all()
            torch.testing.assert_close(means[2, -1], raw[2, -1])
            compact = torch.tensor([[1, 2, 1]])
            torch.testing.assert_close(
                means[3, [0, 2, 4]],
                model._prefix_mean(model.item_emb(compact),
                                   compact.ne(0).unsqueeze(-1))[0],
            )
            changed_raw = raw.clone()
            changed_raw[:, -1] += 3.0
            torch.testing.assert_close(
                means[:, :-1], model._prefix_mean(changed_raw, keep)[:, :-1]
            )
            # Two independent embedding coordinates expose the actual weight
            # ratio of interactions separated by ten observed updates.
            probe = torch.zeros(1, 13, 8)
            probe[0, 0, 0] = 1.0
            probe[0, -1, 1] = 1.0
            probe_keep = torch.ones(1, 13, 1, dtype=torch.bool)
            probe_keep[0, 5:7] = False
            probe[0, 5:7] = 7.0
            pooled = model._prefix_mean(probe, probe_keep)
            torch.testing.assert_close(
                pooled[0, -1, 0] / pooled[0, -1, 1], torch.tensor(0.5)
            )
            differentiable_raw = raw.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                pooled_for_grad = model._prefix_mean(differentiable_raw, keep)
                pooled_for_grad.square().sum().backward()
            assert torch.isfinite(differentiable_raw.grad).all()
            assert differentiable_raw.grad[keep.squeeze(-1)].abs().sum() > 0
            torch.testing.assert_close(
                differentiable_raw.grad[~keep.squeeze(-1)],
                torch.zeros_like(differentiable_raw.grad[~keep.squeeze(-1)]),
            )
            features = model(seq)
            assert torch.isfinite(features).all()
            torch.testing.assert_close(
                features[~keep.squeeze(-1)],
                torch.zeros_like(features[~keep.squeeze(-1)]),
            )
            changed = seq.clone()
            changed[0, -1] = 4
            torch.testing.assert_close(features[:, :-1], model(changed)[:, :-1])
            model.item_emb.weight[0].zero_()
        pos = torch.tensor([
            [0, 0, 2, 3, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [2, 0, 1, 0, 3],
        ])
        negative = torch.arange(144).remainder(8).add(5).repeat(seq.shape[0], 1)
        model.ranking_loss(seq, pos, negative).backward()
        for embedding in (model.item_emb, model.output_emb):
            assert embedding.weight.grad is not None
            assert torch.isfinite(embedding.weight.grad).all()
            assert embedding.weight.grad[1:].abs().sum() > 0
            assert torch.count_nonzero(embedding.weight.grad[0]) == 0
        for branch in (model.preference_proj, model.preference_norm, model.gate):
            for parameter in branch.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
            assert sum(p.grad.abs().sum().item() for p in branch.parameters()) > 0
        # These deterministic loss checks remain inside fork_rng, preserving
        # the training RNG stream and the existing preference diagnostics.
        valid = pos.ne(0)
        d = valid.flip(1).long().cumsum(1).flip(1) - valid.long()
        weights = torch.exp2(-d.float() / 20.0) * valid.float()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        torch.testing.assert_close(
            weights.sum(1), valid.any(1).float()
        )
        assert torch.count_nonzero(weights[~valid]) == 0
        torch.testing.assert_close(weights[2, -1], torch.tensor(1.0))
        torch.testing.assert_close(
            d[3, valid[3]], torch.tensor([2, 1, 0])
        )
        # Twenty subsequent observed targets halve a transition's weight,
        # even when padding separates some of those targets.
        probe_valid = torch.ones(1, 23, dtype=torch.bool)
        probe_valid[0, [5, 13]] = False
        probe_d = (
            probe_valid.flip(1).long().cumsum(1).flip(1)
            - probe_valid.long()
        )
        probe_weights = torch.exp2(-probe_d.float() / 20.0) * probe_valid.float()
        probe_weights /= probe_weights.sum(1, keepdim=True)
        torch.testing.assert_close(
            probe_weights[0, -1] / probe_weights[0, 0], torch.tensor(2.0)
        )
        # Independent reference: compute CE separately for each target, then
        # average normalized, exponentially weighted losses over nonempty rows.
        with torch.no_grad():
            features = model(seq)
            row_losses = []
            for row in range(pos.shape[0]):
                indices = pos[row].nonzero(as_tuple=True)[0].tolist()
                if not indices:
                    continue
                numerator = features.new_zeros((), dtype=torch.float32)
                denominator = 0.0
                for order, column in enumerate(indices):
                    mining_ids = negative[row, :128]
                    mining_scores = (
                        model.output_emb(mining_ids) * features[row, column]
                    ).sum(-1) + model.item_bias(mining_ids).squeeze(-1)
                    # Independent full sort rather than the implementation's topk.
                    order_ids = torch.argsort(mining_scores, descending=True)[:16]
                    candidates = torch.cat([
                        pos[row, column:column + 1], mining_ids[order_ids],
                        negative[row, 128:],
                    ])
                    reference_logits = (
                        model.output_emb(candidates) * features[row, column]
                    ).sum(-1) + model.item_bias(candidates).squeeze(-1)
                    target_loss = F.cross_entropy(
                        reference_logits.float().unsqueeze(0),
                        torch.zeros(1, dtype=torch.long),
                    )
                    weight = 2.0 ** (-(len(indices) - order - 1) / 20.0)
                    numerator = numerator + weight * target_loss
                    denominator += weight
                row_losses.append(numerator / denominator)
            expected_loss = torch.stack(row_losses).mean()
        model.zero_grad(set_to_none=True)
        captured_features = []
        def capture_features(module, inputs, output):
            output.retain_grad()
            captured_features.append(output)
        handle = model.register_forward_hook(capture_features)
        try:
            actual_loss = model.ranking_loss(seq, pos, negative)
        finally:
            handle.remove()
        assert len(captured_features) == 1
        assert torch.isfinite(actual_loss)
        torch.testing.assert_close(actual_loss, expected_loss)
        assert actual_loss.dtype == torch.float32
        actual_loss.backward()
        feature_grad = captured_features[0].grad
        assert torch.isfinite(feature_grad).all()
        assert feature_grad[valid].abs().sum() > 0
        assert torch.count_nonzero(feature_grad[~valid]) == 0
        for branch in (model.preference_proj, model.preference_norm, model.gate):
            for parameter in branch.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
            assert sum(p.grad.abs().sum().item() for p in branch.parameters()) > 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        model.zero_grad(set_to_none=True)
        empty_loss = model.ranking_loss(seq, torch.zeros_like(pos), negative)
        assert empty_loss.requires_grad
        torch.testing.assert_close(empty_loss, torch.tensor(0.0))
        empty_loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
                assert torch.count_nonzero(parameter.grad) == 0
    print('Untied-output checks passed: identical initialization, independent '
          'storage, finite context/output gradients, output-table mining and '
          'prediction; host validation improvement remains unmeasured', flush=True)
    print('Mixed-negative checks passed: complete-history exclusion, independent '
          '128/16 draws with replacement, position-specific biased top-16, '
          '33-way CE, selected embedding/bias/feature gradients', flush=True)
    print('Loss aggregation checks passed: padding, singleton/empty rows, '
          'twenty-target half-life, equal row mass, reference, and gradients; '
          'host validation improvement remains unmeasured', flush=True)
    print('Preference checks passed: exponential weights, empty/singleton/repeated '
          'histories, padding, ten-interaction half-life, causality, and gradients',
          flush=True)


def verify_generated_predictor():
    # Compare independent strict checkpoint reloads through the public routing API.
    spec = importlib.util.spec_from_file_location(
        'generated_predictor_check', os.path.join(WORKING, 'predict.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._ensure_loaded()
    device = module._device
    with torch.random.fork_rng(devices=[]):
        for name in DATASETS:
            with open(os.path.join(WORKING, name + '_meta.pkl'), 'rb') as f:
                meta = pickle.load(f)
            assert meta.get('untied_output') is True
            reference = SASRec(**meta).to(device)
            reference.load_state_dict(
                torch.load(os.path.join(WORKING, name + '_model.pt'),
                           map_location=device, weights_only=True),
                strict=True,
            )
            reference.eval()
            assert (reference.item_emb.weight.data_ptr()
                    != reference.output_emb.weight.data_ptr())
            torch.testing.assert_close(
                reference.output_emb.weight,
                module._models[name].output_emb.weight,
            )
            train, valid, itemnum = load_dataset(name)
            sequences, candidates = make_validation(
                train, valid, itemnum, meta['maxlen']
            )
            with torch.inference_mode():
                expected = reference.predict(
                    sequences[:3].to(device), candidates[:3].to(device)
                ).float().cpu()
            actual = []
            for seq, cand in zip(sequences[:3], candidates[:3]):
                history = seq[seq.ne(0)].tolist()
                actual.append(module.predict(
                    0, history, cand.tolist(), dataset_name=name
                ))
            torch.testing.assert_close(
                torch.tensor(actual), expected, rtol=1e-4, atol=1e-5
            )
            del reference
    module._models.clear()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('Generated predictor scores agree for all four categories', flush=True)


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    verify_preference_branch()
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reserve substantial time below the nine-hour hard limit and guarantee
    # that every category receives its own training allocation.
    start = time.monotonic()
    overall_deadline = start + 8.5 * 3600
    scores = []
    for i, name in enumerate(DATASETS):
        now = time.monotonic()
        category_deadline = now + max(1.0, overall_deadline - now) / (len(DATASETS) - i)
        scores.append(train_one(name, device, category_deadline))
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(PREDICTOR)
    verify_generated_predictor()
    score = float(np.mean(scores))
    # These are local validation comparisons; the unchanged host reports its
    # independently evaluated macro and category metrics after execution.
    print(
        f"Local validation macro_HR@10={score:.5f} "
        f"immediate_reference=0.4838 delta={score - 0.4838:+.5f} "
        f"historical_best=0.5375 delta_to_best={score - 0.5375:+.5f}",
        flush=True,
    )
    reference_scores = dict(zip(DATASETS, [0.520, 0.535, 0.380, 0.500]))
    for name, category_score in zip(DATASETS, scores):
        print(
            f"{name} selected_validation_HR@10={category_score:.5f} "
            f"reference={reference_scores[name]:.5f} "
            f"delta={category_score - reference_scores[name]:+.5f}",
            flush=True,
        )
    electronics = scores[DATASETS.index('Electronics')]
    print(
        f"Electronics local validation deltas: "
        f"vs_previous={electronics - 0.380:+.5f} "
        f"vs_baseline={electronics - 0.420:+.5f}",
        flush=True,
    )
    with open(os.path.join(WORKING, "validation_metrics.json"), "w") as f:
        json.dump(
            {
                "protocol": {
                    "seed": 42,
                    "users_per_category": 200,
                    "unique_negatives": 99,
                    "metric": "HR@10",
                },
                "per_dataset": dict(zip(DATASETS, scores)),
                "macro_HR@10": score,
            },
            f,
            indent=2,
        )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
