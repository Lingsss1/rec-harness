from __future__ import annotations
import os
import math
import time
import random
import json
import importlib.util
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
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
BATCHES = int(os.environ.get("SASREC_BATCHES", "200"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))


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
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_data(ds):
    train = read_split(os.path.join(DATA_DIR, ds + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, ds + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    n = (
        int(stats[ds]["itemnum"])
        if ds in stats
        else max(
            max(items) for items in list(train.values()) + list(valid.values()) if items
        )
    )
    return train, valid, n


def pack_history(histories, length):
    a = np.zeros((len(histories), length), dtype=np.int64)
    for i, h in enumerate(histories):
        h = h[-length:]
        if len(h):
            a[i, -len(h) :] = h
    return a


class PointWiseFeedForward(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum,
        hidden_units=64,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.5,
        maxlen=128,
    ):
        super().__init__()
        self.maxlen = maxlen
        self.d = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=1.0)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.drop = nn.Dropout(dropout_rate)
        self.an = nn.ModuleList(
            [nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)]
        )
        self.att = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_units, num_heads, dropout=dropout_rate, batch_first=True
                )
                for _ in range(num_blocks)
            ]
        )
        self.fn = nn.ModuleList(
            [nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)]
        )
        self.ff = nn.ModuleList(
            [
                PointWiseFeedForward(hidden_units, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.last = nn.LayerNorm(hidden_units, eps=1e-8)

    def log2feats(self, seq):
        keep = seq.ne(0).unsqueeze(-1)
        x = (
            self.drop(
                self.item_emb(seq) * math.sqrt(self.d)
                + self.pos_emb(torch.arange(seq.shape[1], device=seq.device))
            )
            * keep
        )
        mask = torch.ones(
            seq.shape[1], seq.shape[1], device=seq.device, dtype=torch.bool
        ).triu(1)
        for an, att, fn, ff in zip(self.an, self.att, self.fn, self.ff):
            q = an(x)
            y = att(q, x, x, attn_mask=mask, need_weights=False)[0]
            x = ff(fn(q + y)) * keep
        return self.last(x)

    def forward(self, seq, pos=None, neg=None):
        x = self.log2feats(seq)
        if pos is None or neg is None:
            return x
        # Gather active positions before embedding the 16 competing negatives.
        keep = pos.ne(0)
        features = x[keep]
        positive_ids = pos[keep]
        negative_ids = neg[keep]
        a = (features * self.item_emb(positive_ids)).sum(-1)
        b = (features.unsqueeze(1) * self.item_emb(negative_ids)).sum(-1)
        return a, b

    def predict(self, seq, candidates):
        return (
            self.item_emb(candidates) * self.log2feats(seq)[:, -1:].contiguous()
        ).sum(-1)


class ResidualVAE(nn.Module):
    def __init__(self, itemnum, hidden=256, latent=64, dropout=0.5):
        super().__init__()
        self.itemnum = itemnum
        self.dropout = dropout
        self.encoder = nn.EmbeddingBag(itemnum + 1, hidden, mode="sum", padding_idx=0)
        # Blend a conservative temporal branch into the original uniform encoder.
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.enc_bias = nn.Parameter(torch.zeros(hidden))
        self.posterior = nn.Linear(hidden, 2 * latent)
        self.decoder_hidden = nn.Linear(latent, hidden)
        self.output = nn.Linear(hidden, itemnum + 1)
        nn.init.normal_(self.encoder.weight, std=0.02)
        nn.init.xavier_normal_(self.output.weight)
        with torch.no_grad():
            self.encoder.weight[0].zero_()

    def representation(self, ids, offsets, counts, recency_weights=None):
        if recency_weights is None:
            recency_weights = ids.ne(0).to(self.encoder.weight.dtype)
        # Both denominators describe the original inputs, before item dropout.
        uniform_norm = counts.clamp_min(1).sqrt().unsqueeze(-1)
        lengths = torch.diff(torch.cat([offsets, offsets.new_tensor([ids.numel()])]))
        bag_index = torch.repeat_interleave(
            torch.arange(offsets.numel(), device=ids.device), lengths
        )
        squared_norm = recency_weights.new_zeros(offsets.numel()).scatter_add(
            0, bag_index, recency_weights.square()
        )
        temporal_norm = squared_norm.clamp_min(
            torch.finfo(recency_weights.dtype).tiny
        ).sqrt().unsqueeze(-1)
        # One mask drops the same items from both sums, with inverted scaling.
        if self.training:
            ids = ids * (torch.rand(ids.shape, device=ids.device) >= self.dropout)
        uniform = self.encoder(ids, offsets) / uniform_norm
        recency = self.encoder(
            ids, offsets, per_sample_weights=recency_weights
        ) / temporal_norm
        if self.training:
            uniform = uniform / (1.0 - self.dropout)
            recency = recency / (1.0 - self.dropout)
        gate = self.gate_logit.sigmoid()
        encoded = (1.0 - gate) * uniform + gate * recency
        encoded = torch.tanh(encoded + self.enc_bias)
        mu, logvar = self.posterior(encoded).chunk(2, -1)
        logvar = logvar.clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        h = torch.tanh(self.decoder_hidden(z)) + encoded
        return h, mu, logvar

    def forward(self, ids, offsets, counts, recency_weights=None):
        h, mu, logvar = self.representation(
            ids, offsets, counts, recency_weights=recency_weights
        )
        logits = self.output(h)
        logits = logits.masked_fill(
            torch.arange(self.itemnum + 1, device=ids.device)[None] == 0, -1e9
        )
        return logits, mu, logvar

    def predict(self, ids, offsets, counts, candidates, recency_weights=None):
        h, _, _ = self.representation(
            ids, offsets, counts, recency_weights=recency_weights
        )
        return (self.output.weight[candidates] * h.unsqueeze(1)).sum(
            -1
        ) + self.output.bias[candidates]


def bag_tensors(histories, device, return_weights=False):
    bags = [sorted(set(h)) or [0] for h in histories]
    lengths = [len(h) for h in bags]
    ids = torch.tensor([i for h in bags for i in h], dtype=torch.long, device=device)
    offsets = torch.tensor(
        np.cumsum([0] + lengths[:-1]), dtype=torch.long, device=device
    )
    counts = torch.tensor(
        [len(set(h) - {0}) for h in bags], dtype=torch.float32, device=device
    )
    if return_weights:
        weights = []
        for history, bag in zip(histories, bags):
            # Record the last position before deduplication. Inputs for prefix
            # tasks have already been cut; future items never affect distances.
            latest = {item: position for position, item in enumerate(history) if item}
            weights.extend(
                2.0 ** (-(len(history) - 1 - latest[item]) / 20.0)
                if item else 0.0
                for item in bag
            )
        recency_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return ids, offsets, counts, recency_weights
    return ids, offsets, counts


def ranks(scores, candidates):
    # Stable item-ID tie breaking, independent of candidate position.
    order = np.lexsort((np.asarray(candidates), -np.asarray(scores)))
    result = np.empty(len(order), dtype=np.float64)
    result[order] = np.arange(1, len(order) + 1)
    return result


def rank_fusion(a, b, candidates):
    return 0.5 / (60.0 + ranks(a, candidates)) + 0.5 / (60.0 + ranks(b, candidates))


@torch.no_grad()
def expert_scores(model, kind, histories, candidates, device):
    model.eval()
    c = torch.tensor(candidates, dtype=torch.long, device=device)
    if kind == "sas":
        seq = torch.tensor(pack_history(histories, model.maxlen), device=device)
        out = model.predict(seq, c)
    else:
        ids, offsets, counts, recency_weights = bag_tensors(
            histories, device, return_weights=True
        )
        out = model.predict(
            ids, offsets, counts, c, recency_weights=recency_weights
        )
    return out.float().cpu().numpy()


def validation_cases(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    result = []
    for u in users:
        if not valid[u]:
            continue
        excluded = set(train[u]) | set(valid[u])
        if itemnum - len(excluded) < 99:
            raise ValueError("Fewer than 99 available unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            i = rng.randint(1, itemnum)
            if i not in excluded:
                negatives.add(i)
        result.append((train[u], [valid[u][0]] + list(negatives)))
    if not result:
        raise ValueError("No validation cases")
    return result


def metric(score_rows, cases):
    hit, ndcg = 0.0, 0.0
    for score, (_, candidates) in zip(score_rows, cases):
        rank = int(ranks(score, candidates)[0])
        if rank <= 10:
            hit += 1
            ndcg += 1.0 / math.log2(rank + 1)
    return hit / len(cases), ndcg / len(cases)


def evaluate(model, kind, cases, device):
    rows = []
    for start in range(0, len(cases), 32):
        chunk = cases[start : start + 32]
        rows.extend(
            expert_scores(
                model, kind, [x[0] for x in chunk], [x[1] for x in chunk], device
            )
        )
    return metric(rows, cases), np.asarray(rows)


class HistoryDataset(Dataset):
    def __init__(self, histories):
        self.histories = histories

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, i):
        return self.histories[i]


def collate_histories(batch):
    return batch


def check_temporal_encoder():
    # CPU-only, isolated Torch RNG; no Python/NumPy draws or training-state changes.
    import io

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        device = torch.device("cpu")
        histories = [[3, 1, 3, 2], [], [0, 0], [1, 2, 3]]
        ids, offsets, counts, weights = bag_tensors(
            histories, device, return_weights=True
        )
        legacy = bag_tensors(histories, device)
        for original, weighted in zip(legacy, (ids, offsets, counts)):
            torch.testing.assert_close(original, weighted)
        assert ids.tolist() == [1, 2, 3, 0, 0, 1, 2, 3]
        torch.testing.assert_close(
            weights[:3], torch.tensor([2 ** (-2 / 20), 1.0, 2 ** (-1 / 20)])
        )
        assert weights[3:5].eq(0).all().item()
        assert counts.tolist() == [3.0, 0.0, 0.0, 3.0]
        # Prefix distances end at the cut, including when a later duplicate exists.
        full = [3, 1, 2, 3, 4]
        prefix = bag_tensors([full[:3]], device, return_weights=True)
        torch.testing.assert_close(
            prefix[3], torch.tensor([2 ** (-1 / 20), 1.0, 2 ** (-2 / 20)])
        )
        model = ResidualVAE(12, hidden=16, latent=4, dropout=0.5)
        captured = []
        hook = model.posterior.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        try:
            model.train()
            # Replay the sole item mask independently, before posterior sampling.
            state = torch.get_rng_state()
            keep = torch.rand(ids.shape) >= model.dropout
            torch.set_rng_state(state)
            differentiable_weights = weights.clone().requires_grad_(True)
            logits, mu, logvar = model(
                ids, offsets, counts, recency_weights=differentiable_weights
            )
            embeddings = model.encoder.weight[ids] * keep[:, None]
            uniforms, temporals = [], []
            ends = offsets.tolist()[1:] + [ids.numel()]
            for start, end, count in zip(offsets.tolist(), ends, counts):
                e, w = embeddings[start:end], weights[start:end]
                uniforms.append(e.sum(0) / count.clamp_min(1).sqrt())
                temporals.append(
                    (e * w[:, None]).sum(0)
                    / w.square().sum().clamp_min(torch.finfo(w.dtype).tiny).sqrt()
                )
            uniform = torch.stack(uniforms) / (1 - model.dropout)
            temporal = torch.stack(temporals) / (1 - model.dropout)
            gate = model.gate_logit.sigmoid()
            expected = torch.tanh(
                (1 - gate) * uniform + gate * temporal + model.enc_bias
            )
            torch.testing.assert_close(captured[-1], expected)
            assert torch.isfinite(logits).all().item()
            assert torch.isfinite(captured[-1]).all().item()
            torch.testing.assert_close(captured[-1][1:3], torch.zeros(2, 16))
            loss = logits[:, 1:].square().mean() + mu.square().mean() + logvar.square().mean()
            loss.backward()
            for gradient in (
                model.encoder.weight.grad, model.gate_logit.grad,
                differentiable_weights.grad,
            ):
                assert gradient is not None and torch.isfinite(gradient).all().item()

            model.eval()
            with torch.no_grad():
                model.gate_logit.fill_(-80)
                model.representation(ids, offsets, counts, recency_weights=weights)
                original = model.encoder(ids, offsets) / counts.clamp_min(1).sqrt()[:, None]
                torch.testing.assert_close(
                    captured[-1], torch.tanh(original + model.enc_bias)
                )
                model.gate_logit.fill_(20)
                ordered = bag_tensors(
                    [[1, 2, 3], [3, 2, 1]], device, return_weights=True
                )
                model.representation(*ordered[:3], recency_weights=ordered[3])
                assert (captured[-1][0] - captured[-1][1]).abs().max().item() > 1e-7
                candidates = torch.tensor([[1, 2, 3]] * len(histories))
                expected_scores = model.predict(
                    ids, offsets, counts, candidates, recency_weights=weights
                )
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = ResidualVAE(12, hidden=16, latent=4, dropout=0.5).eval()
                restored.load_state_dict(
                    torch.load(buffer, map_location="cpu", weights_only=True), strict=True
                )
                torch.testing.assert_close(
                    restored.predict(
                        ids, offsets, counts, candidates, recency_weights=weights
                    ), expected_scores,
                )
        finally:
            hook.remove()


def train_vae(ds, train, n, cases, device, deadline):
    seed_all(42)
    config = dict(itemnum=n, hidden=256, latent=64, dropout=0.5)
    model = ResidualVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    histories = [h for h in train.values() if h]
    loader = DataLoader(
        HistoryDataset(histories),
        batch_size=128,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_histories,
        generator=torch.Generator().manual_seed(42),
    )
    rng = np.random.RandomState(42)
    path = os.path.join(WORK, ds + "_vae.pt")
    best, stale, step = (-1.0, -1.0), 0, 0
    for epoch in range(1, min(EPOCHS, 150) + 1):
        model.train()
        loss_sum, batches = 0.0, 0
        for batch in loader:
            inputs, targets = [], []
            for h in batch:
                # A shared residual encoder learns both complete preferences and
                # a chronological prefix-to-future reconstruction task.
                if len(h) > 2 and rng.rand() < 0.30:
                    cut = rng.randint(1, len(h))
                    inputs.append(h[:cut])
                    targets.append(sorted(set(h[cut:])))
                else:
                    inputs.append(h)
                    targets.append(sorted(set(h)))
            ids, offsets, counts, recency_weights = bag_tensors(
                inputs, device, return_weights=True
            )
            logits, mu, logvar = model(
                ids, offsets, counts, recency_weights=recency_weights
            )
            lengths = [len(t) for t in targets]
            rows = torch.repeat_interleave(
                torch.arange(len(batch), device=device),
                torch.tensor(lengths, device=device),
            )
            columns = torch.tensor(
                [i for t in targets for i in t], dtype=torch.long, device=device
            )
            logprob = logits.log_softmax(-1)
            # Normalized multinomial reconstruction keeps long histories from
            # dominating the bridge task.
            weights = torch.repeat_interleave(
                1.0 / torch.tensor(lengths, device=device, dtype=torch.float32),
                torch.tensor(lengths, device=device),
            )
            rec = -(logprob[rows, columns] * weights).sum() / len(batch)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
            anneal = min(0.10, step / 20000.0)
            loss = rec + anneal * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            batches += 1
            loss_sum += loss.item()
            if batches >= BATCHES or time.monotonic() >= deadline:
                break
        value, _ = evaluate(model, "vae", cases, device)
        print(
            f"{ds} VAE epoch={epoch} loss={loss_sum/max(batches,1):.4f} HR@10={value[0]:.4f}",
            flush=True,
        )
        if value > best:
            best, stale = value, 0
            torch.save(
                {"config": config, "state": model.state_dict(),
                 "encoding_version": "latest_position_half_life_20_v1"},
                path,
            )
        else:
            stale += 1
        if stale >= 20 or time.monotonic() >= deadline:
            break
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)["state"]
    )
    return model.eval()


def sample_sas_negatives(pos, indices, seen, n, rng):
    # Independent uniform draws with replacement; exclude complete train history.
    # Padding receives no draws and remains zero.
    neg = np.zeros(pos.shape + (16,), dtype=np.int64)
    for row, idx in enumerate(indices):
        columns = np.flatnonzero(pos[row])
        if not len(columns):
            continue
        excluded = np.asarray(tuple(seen[idx]), dtype=np.int64)
        draws = rng.randint(1, n + 1, size=(len(columns), 16))
        invalid = np.isin(draws, excluded)
        while invalid.any():
            draws[invalid] = rng.randint(1, n + 1, size=int(invalid.sum()))
            invalid = np.isin(draws, excluded)
        neg[row, columns] = draws
    return neg


def check_sampled_softmax():
    # Use independent NumPy state and restore CPU Torch state so these checks
    # cannot perturb training initialization, dropout, or sampling.
    rng = np.random.RandomState(123)
    pos = np.array([[0, 2, 3], [0, 0, 5]], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    seen = [{1, 2, 3, 7}, {4, 5, 6, 8}]
    neg = sample_sas_negatives(pos, indices, seen, 12, rng)
    assert neg.shape == (2, 3, 16)
    assert np.all(neg[pos == 0] == 0)
    for row, idx in enumerate(indices):
        active = neg[row, pos[row] != 0]
        assert np.all((active >= 1) & (active <= 12))
        assert not np.isin(active, list(seen[idx])).any()

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        model = SASRec(12, hidden_units=8, num_blocks=1,
                       num_heads=1, dropout_rate=0.0, maxlen=3).eval()
        seq_t = torch.tensor([[0, 1, 2], [0, 0, 4]])
        pos_t = torch.tensor(pos)
        neg_t = torch.tensor(neg)
        a, b = model(seq_t, pos_t, neg_t)
        assert a.shape == (3,) and b.shape == (3, 16)
        logits = torch.cat([a[:, None], b], dim=1).float()
        loss = nn.functional.cross_entropy(
            logits, torch.zeros(a.shape[0], dtype=torch.long)
        )
        # Independently score all 17 candidates and form log-normalized loss.
        features = model.log2feats(seq_t)[pos_t.ne(0)]
        candidates = torch.cat(
            [pos_t[pos_t.ne(0)][:, None], neg_t[pos_t.ne(0)]], dim=1
        )
        reference_logits = torch.einsum(
            "nd,nkd->nk", features, model.item_emb(candidates)
        ).float()
        reference = (
            torch.logsumexp(reference_logits, dim=1) - reference_logits[:, 0]
        ).mean()
        torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(loss).item()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all().item() for g in gradients)


def train_sas(ds, train, n, cases, device, deadline):
    check_sampled_softmax()
    seed_all(42)
    config = dict(
        itemnum=n,
        hidden_units=int(os.environ.get("SASREC_HIDDEN", "64")),
        num_blocks=int(os.environ.get("SASREC_BLOCKS", "2")),
        num_heads=int(os.environ.get("SASREC_HEADS", "1")),
        dropout_rate=float(os.environ.get("SASREC_DROPOUT", "0.5")),
        maxlen=MAXLEN,
    )
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(os.environ.get("SASREC_LR", "0.001")),
        betas=(0.9, 0.98),
        weight_decay=float(os.environ.get("SASREC_WD", "0")),
    )
    histories = [h for h in train.values() if len(h) >= 2 and len(set(h)) < n]
    if not histories:
        raise ValueError("No eligible SASRec training histories")
    seqs = pack_history([h[:-1] for h in histories], MAXLEN)
    positives = pack_history([h[1:] for h in histories], MAXLEN)
    seen = [set(h) for h in histories]
    rng = np.random.RandomState(42)
    path = os.path.join(WORK, ds + "_sas.pt")
    best, stale = (-1.0, -1.0), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, steps = 0.0, 0
        for _ in range(BATCHES):
            indices = rng.randint(len(histories), size=BATCH)
            seq = seqs[indices]
            pos = positives[indices]
            neg = sample_sas_negatives(pos, indices, seen, n, rng)
            seq_t = torch.tensor(seq, device=device)
            pos_t = torch.tensor(pos, device=device)
            neg_t = torch.tensor(neg, device=device)
            a, b = model(seq_t, pos_t, neg_t)
            # Temperature-one competition, uniformly averaged over active positions.
            loss = nn.functional.cross_entropy(
                torch.cat([a[:, None], b], dim=1).float(),
                torch.zeros(a.shape[0], dtype=torch.long, device=device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
            steps += 1
            if time.monotonic() >= deadline:
                break
        if epoch % 5 == 0 or epoch == 1 or time.monotonic() >= deadline:
            value, _ = evaluate(model, "sas", cases, device)
            print(
                f"{ds} SAS epoch={epoch} loss={total/max(steps,1):.4f} HR@10={value[0]:.4f}",
                flush=True,
            )
            if value > best:
                best, stale = value, 0
                torch.save({"config": config, "state": model.state_dict()}, path)
            else:
                stale += 1
            if stale >= 20:
                break
        if time.monotonic() >= deadline:
            break
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)["state"]
    )
    return model.eval()


class Predictor:
    def __init__(self, working_dir=WORK):
        self.working_dir = working_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def load(self, ds):
        if ds not in self.models:
            experts = []
            for kind, cls in [("sas", SASRec), ("vae", ResidualVAE)]:
                blob = torch.load(
                    os.path.join(self.working_dir, ds + "_" + kind + ".pt"),
                    map_location=self.device,
                    weights_only=True,
                )
                if kind == "vae" and blob.get("encoding_version") != (
                    "latest_position_half_life_20_v1"
                ):
                    raise ValueError("Incompatible VAE temporal encoding checkpoint")
                model = cls(**blob["config"]).to(self.device)
                model.load_state_dict(blob["state"], strict=True)
                experts.append(model.eval())
            self.models[ds] = experts
        return self.models[ds]

    def predict(self, user_id, history, candidates, *, dataset_name=None):
        ds = dataset_name if dataset_name is not None else DATASETS[0]
        if ds not in DATASETS:
            raise ValueError("Unknown dataset: " + str(ds))
        sas, vae = self.load(ds)
        if len(candidates) == 0:
            return []
        a = expert_scores(sas, "sas", [history], [candidates], self.device)[0]
        b = expert_scores(vae, "vae", [history], [candidates], self.device)[0]
        return rank_fusion(a, b, candidates).tolist()


def write_predict_script():
    # Import the exact architecture and preprocessing implementation used above.
    script = """import importlib.util
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('_residual_bridge_train', os.path.join(_HERE, '..', 'train.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
# Predictor strictly checks temporal checkpoint metadata and restores gate_logit.
# Importing train.py shares the exact bag transformation and weighted encoder.
_predictor = _mod.Predictor(_HERE)
def predict(user_id, history, candidates, *, dataset_name=None):
    return _predictor.predict(user_id, history, candidates, dataset_name=dataset_name)
"""
    with open(os.path.join(WORK, "predict.py"), "w") as f:
        f.write(script)


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORK, exist_ok=True)
    check_temporal_encoder()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Wall-clock cap leaves substantial room below the nine-hour hard limit.
    total_seconds = min(float(os.environ.get("TRAIN_TIME_LIMIT", "24000")), 28000.0)
    end = time.monotonic() + total_seconds
    scores = []
    write_predict_script()
    print(
        "Validation: supplied train/valid splits; seed=42; up to 200 users/category; "
        "99 unique unseen negatives; HR@10=mean(target rank<=10); macro-average over four categories. "
        "Prior host HR@10: macro=0.6062; Movies_and_TV=0.645; "
        "Industrial_and_Scientific=0.650; Electronics=0.455; CDs_and_Vinyl=0.675. "
        "Current host results remain pending; local validation does not establish a host gain.",
        flush=True,
    )
    for index, ds in enumerate(DATASETS):
        train, valid, n = load_data(ds)
        cases = validation_cases(train, valid, n)
        remaining = max(1.0, end - time.monotonic() - 120.0)
        allowance = remaining / (len(DATASETS) - index)
        start = time.monotonic()
        vae = train_vae(ds, train, n, cases, device, start + allowance * 0.48)
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sas = train_sas(ds, train, n, cases, device, start + allowance * 0.96)
        del sas
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Validate via the deployed predictor, one user at a time.
        predictor = Predictor(WORK)
        rows = [predictor.predict(0, h, c, dataset_name=ds) for h, c in cases]
        value = metric(rows, cases)
        scores.append(value[0])
        print(f"{ds} fused HR@10={value[0]:.6f} NDCG@10={value[1]:.6f}", flush=True)
        # Check the generated module using identical deployed inputs.
        spec = importlib.util.spec_from_file_location(
            "_export_check", os.path.join(WORK, "predict.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        h, c = cases[0]
        exported = mod.predict(0, h, c, dataset_name=ds)
        np.testing.assert_allclose(exported, rows[0], rtol=1e-6, atol=1e-8)
        # Compare raw temporal scores too: rank fusion could hide score drift.
        original_vae = predictor.load(ds)[1]
        exported_vae = mod._predictor.load(ds)[1]
        probe_histories = [h, h[::-1], [], [1, 2, 1]]
        probe_candidates = [c] * len(probe_histories)
        np.testing.assert_allclose(
            expert_scores(
                original_vae, "vae", probe_histories, probe_candidates, predictor.device
            ),
            mod._mod.expert_scores(
                exported_vae, "vae", probe_histories, probe_candidates,
                mod._predictor.device,
            ),
            rtol=1e-6, atol=1e-8,
        )
        del mod, predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()