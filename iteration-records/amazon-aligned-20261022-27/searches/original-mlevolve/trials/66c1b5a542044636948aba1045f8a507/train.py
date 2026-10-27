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
HERE = os.path.dirname(os.path.abspath(__file__))
WORKING = os.path.join(HERE, "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
BATCH = int(os.environ.get("VAE_BATCH", "256"))
PATIENCE = 25
SAS_BATCH = int(os.environ.get("SASREC_BATCH_SIZE", "128"))


def seed_training():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def read_split(path):
    result = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as handle:
            stats = json.load(handle)
        nitems = int(stats[name]["itemnum"])
    else:
        nitems = max(max(items) for items in train.values() if items)
        nitems = max(nitems, max((max(v) for v in valid.values() if v), default=0))
    return train, valid, nitems


class Histories(Dataset):
    def __init__(self, train):
        self.rows = [
            np.asarray(sorted(set(items)), dtype=np.int64)
            for items in train.values()
            if items
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if len(row) == 1:
            return row.copy(), row.copy()
        count = min(len(row) - 1, max(1, int(math.floor(0.2 * len(row)))))
        hidden = np.zeros(len(row), dtype=bool)
        hidden[np.random.choice(len(row), count, replace=False)] = True
        return row[~hidden], row[hidden]


def pack_rows(rows):
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    ids = np.concatenate(rows) if offsets[-1] else np.empty(0, dtype=np.int64)
    owners = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
    return (torch.from_numpy(ids), torch.from_numpy(offsets), torch.from_numpy(owners))


def pack_training_rows(rows):
    visible, targets = zip(*rows)
    ids, offsets, owners = pack_rows(visible)
    target_ids, _, target_owners = pack_rows(targets)
    return ids, offsets, owners, target_ids, target_owners


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MultVAE(nn.Module):
    def __init__(self, itemnum, hidden=256, latent=64, dropout=0.5,
                 residual=False):
        super().__init__()
        self.itemnum = itemnum
        self.dropout = dropout
        self.residual = residual
        self.encoder_items = nn.EmbeddingBag(
            itemnum + 1, hidden, mode="sum", include_last_offset=True, padding_idx=0
        )
        self.encoder_bias = nn.Parameter(torch.zeros(hidden))
        self.posterior = nn.Linear(hidden, 2 * latent)
        self.decoder_hidden = nn.Linear(latent, hidden)
        self.decoder = nn.Linear(hidden, itemnum + 1)
        nn.init.normal_(self.encoder_items.weight, std=0.02)
        with torch.no_grad():
            self.encoder_items.weight[0].zero_()
        for layer in [self.posterior, self.decoder_hidden, self.decoder]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _encode_with_hidden(self, ids, offsets):
        lengths = offsets[1:] - offsets[:-1]
        weights = torch.repeat_interleave(lengths.clamp_min(1).float().rsqrt(), lengths)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        h = torch.tanh(
            self.encoder_items(ids, offsets, per_sample_weights=weights)
            + self.encoder_bias
        )
        mu, logvar = self.posterior(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-12.0, 8.0), h

    def encode(self, ids, offsets):
        # Preserve the public two-value contract.
        mu, logvar, _ = self._encode_with_hidden(ids, offsets)
        return mu, logvar

    def representation(self, ids, offsets):
        # Reuse one encoded history and one dropout realization for both paths.
        mu, logvar, encoder_hidden = self._encode_with_hidden(ids, offsets)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        h = torch.tanh(self.decoder_hidden(z))
        if self.residual:
            h = h + encoder_hidden
        return h, mu, logvar

    def forward(self, ids, offsets, owners, anneal,
                target_ids=None, target_owners=None):
        h, mu, logvar = self.representation(ids, offsets)
        logits = self.decoder(h).float()
        logits[:, 0] = -1e9
        logprobs = F.log_softmax(logits, dim=-1)
        if (target_ids is None) != (target_owners is None):
            raise ValueError("Supply both reconstruction target tensors")
        if target_ids is None:
            target_ids, target_owners = ids, owners
        # Only reconstruction indexing uses hidden IDs. Both representation
        # paths above see visible items exclusively. Fewer targets increase the
        # effective KL contribution; retain the original annealing unchanged.
        nll = -logprobs[target_owners, target_ids].sum() / h.shape[0]
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return nll + anneal * kl

    def predict(self, ids, offsets, candidates):
        h, _, _ = self.representation(ids, offsets)
        weights = F.embedding(candidates, self.decoder.weight)
        biases = self.decoder.bias[candidates]
        return (weights * h.unsqueeze(1)).sum(-1) + biases


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden, hidden, 1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        output = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(1, 2))))))
        return inputs + output.transpose(1, 2)


class SASRec(nn.Module):
    def __init__(self, itemnum, maxlen=200, hidden=50, blocks=2,
                 heads=1, dropout=0.5):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.hidden = hidden
        self.item_emb = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.emb_dropout = nn.Dropout(dropout)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden, heads, dropout))
            self.forward_layernorms.append(nn.LayerNorm(hidden, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden, dropout))
        self.last_layernorm = nn.LayerNorm(hidden, eps=1e-8)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def features(self, seq):
        positions = torch.arange(seq.shape[1], device=seq.device)
        h = self.item_emb(seq) * math.sqrt(self.hidden)
        h = self.emb_dropout(h + self.pos_emb(positions).unsqueeze(0))
        padding = seq.eq(0).unsqueeze(-1)
        h = h.masked_fill(padding, 0)
        mask = torch.ones(seq.shape[1], seq.shape[1], dtype=torch.bool,
                          device=seq.device).triu(1)
        for norm, attention, forward_norm, forward in zip(
                self.attention_layernorms, self.attention_layers,
                self.forward_layernorms, self.forward_layers):
            values = h.transpose(0, 1)
            queries = norm(values)
            attended, _ = attention(queries, values, values,
                                    attn_mask=mask, need_weights=False)
            h = (queries + attended).transpose(0, 1)
            h = forward(forward_norm(h)).masked_fill(padding, 0)
        return self.last_layernorm(h)

    def forward(self, seq, pos=None, neg=None):
        h = self.features(seq)
        if pos is None:
            return h
        # Gather active transitions before embedding the 16 alternatives.
        if neg is None or neg.shape != (*pos.shape, 16):
            raise ValueError("Expected 16 negatives per SASRec transition")
        active = pos.ne(0)
        h = h[active]
        return ((h * self.item_emb(pos[active])).sum(-1),
                (h.unsqueeze(1) * self.item_emb(neg[active])).sum(-1))

    def predict(self, seq, candidates):
        h = self.features(seq)[:, -1]
        return (self.item_emb(candidates) * h.unsqueeze(1)).sum(-1)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) > 1]
    if not users:
        raise ValueError("SASRec requires histories with at least two items")
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    negatives = np.zeros((batch_size, maxlen, 16), dtype=np.int64)
    for row in range(batch_size):
        items = user_train[users[np.random.randint(len(users))]]
        excluded = set(items)
        if len(excluded) >= itemnum:
            raise ValueError("No eligible training negatives")
        target = items[-1]
        index = maxlen - 1
        for item in reversed(items[:-1]):
            sequences[row, index] = item
            positives[row, index] = target
            # Independent uniform rejection draws, with replacement.
            for alternative in range(16):
                negative = np.random.randint(1, itemnum + 1)
                while negative in excluded:
                    negative = np.random.randint(1, itemnum + 1)
                negatives[row, index, alternative] = negative
            target = item
            index -= 1
            if index < 0:
                break
    return sequences, positives, negatives


def verify_sasrec_objective():
    # Synthetic CPU checks preserve the seed-42 training streams and consume no
    # training data. Check both padding and exclusion beyond a truncated window.
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            np.random.seed(123)
            torch.manual_seed(123)
            user_train = {1: [1, 2, 3, 4, 5]}
            for maxlen in (3, 6):
                sequences, positives, negatives = sample_batch(
                    user_train, 9, 2, maxlen)
                active = positives != 0
                assert sequences.shape == positives.shape == (2, maxlen)
                assert negatives.shape == (2, maxlen, 16)
                assert np.all(sequences[~active] == 0)
                assert np.all(negatives[~active] == 0)
                assert np.all((negatives[active] >= 1) &
                              (negatives[active] <= 9))
                assert not np.isin(negatives[active], user_train[1]).any()
                expected_length = min(4, maxlen)
                for row in range(2):
                    np.testing.assert_array_equal(
                        sequences[row, active[row]],
                        np.asarray(user_train[1][:-1][-expected_length:]))
                    np.testing.assert_array_equal(
                        positives[row, active[row]],
                        np.asarray(user_train[1][1:][-expected_length:]))
            model = SASRec(itemnum=9, maxlen=6, hidden=8, blocks=1,
                           heads=1, dropout=0.5)
            seq, pos, neg = [torch.as_tensor(x) for x in
                             (sequences, positives, negatives)]
            positive_logits, negative_logits = model(seq, pos, neg)
            count = int(pos.ne(0).sum())
            assert positive_logits.shape == (count,)
            assert negative_logits.shape == (count, 16)
            loss = F.cross_entropy(
                torch.cat([positive_logits[:, None], negative_logits], dim=1).float(),
                torch.zeros(count, dtype=torch.long))
            assert torch.isfinite(loss).item()
            loss.backward()
            for parameter in model.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all().item()
            assert model.item_emb.weight.grad.abs().sum().item() > 0
            assert (model.attention_layers[0].in_proj_weight.grad.abs()
                    .sum().item() > 0)
    finally:
        np.random.set_state(numpy_state)


def chronological_rows(histories, maxlen, device):
    rows = np.zeros((len(histories), maxlen), dtype=np.int64)
    for index, history in enumerate(histories):
        tail = list(history)[-maxlen:]
        if tail:
            rows[index, -len(tail):] = tail
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def descending_ranks(scores):
    # Sorted equal-score runs receive their average zero-based rank.
    size = scores.shape[-1]
    if size == 0:
        return scores.float()
    values, order = torch.sort(scores, dim=-1, descending=True, stable=True)
    positions = torch.arange(size, device=scores.device, dtype=torch.long)
    positions = positions.expand_as(order)
    starts = torch.ones_like(order, dtype=torch.bool)
    ends = torch.ones_like(order, dtype=torch.bool)
    starts[..., 1:] = values[..., 1:] != values[..., :-1]
    ends[..., :-1] = values[..., :-1] != values[..., 1:]
    first = torch.where(starts, positions, torch.zeros_like(positions))
    first = first.cummax(dim=-1).values
    last = torch.where(ends, positions, torch.full_like(positions, size - 1))
    last = last.flip(-1).cummin(dim=-1).values.flip(-1)
    average = (first.float() + last.float()) * 0.5
    return torch.empty_like(average).scatter_(-1, order, average)


def fuse_ranks(vae_rank, sasrec_rank, alpha=0.5):
    if vae_rank.shape != sasrec_rank.shape:
        raise ValueError("Expert candidate shapes differ")
    if not math.isfinite(float(alpha)) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Fusion alpha must be finite and in [0, 1]")
    denominator = max(vae_rank.shape[-1] - 1, 1)
    return -(alpha * vae_rank + (1.0 - alpha) * sasrec_rank) / denominator


def rank_fusion(vae_scores, sasrec_scores, alpha=0.5):
    if vae_scores.shape != sasrec_scores.shape:
        raise ValueError("Expert candidate shapes differ")
    return fuse_ranks(descending_ranks(vae_scores),
                      descending_ranks(sasrec_scores), alpha)


@torch.inference_mode()
def expert_scores(model, packed, candidates, histories, device):
    candidates = candidates.to(device)
    if isinstance(model, tuple):
        vae, sasrec = model
        return rank_fusion(
            expert_scores(vae, packed, candidates, histories, device),
            expert_scores(sasrec, packed, candidates, histories, device))
    if isinstance(model, SASRec):
        return model.predict(chronological_rows(histories, model.maxlen, device),
                             candidates)
    ids, offsets, _ = [x.to(device) for x in packed]
    return model.predict(ids, offsets, candidates)


def validation_batches(train, valid, nitems):
    # Fixed candidates across checkpoints; local RNG never changes training RNG.
    rng = random.Random(42)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    examples = []
    for u in users:
        target = valid[u][0]
        excluded = set(train[u]) | set(valid[u])
        if nitems - len(excluded) < 99:
            raise ValueError("Fewer than 99 eligible unique negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, nitems)
            if item not in excluded:
                negatives.add(item)
        examples.append(
            (
                np.asarray(sorted(set(train[u])), dtype=np.int64),
                [target] + list(negatives),
                list(train[u]),
            )
        )
    if not examples:
        raise ValueError("No eligible validation users")
    batches = []
    for start in range(0, len(examples), 64):
        part = examples[start : start + 64]
        batches.append(
            (
                pack_rows([x[0] for x in part]),
                torch.tensor([x[1] for x in part], dtype=torch.long),
                [x[2] for x in part],
            )
        )
    return batches


@torch.inference_mode()
def evaluate(model, batches, device):
    for member in model if isinstance(model, tuple) else (model,):
        member.eval()
    hits = ndcg = count = 0
    for packed, candidates, histories in batches:
        scores = expert_scores(model, packed, candidates, histories, device)
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        ranks = (order == 0).long().argmax(dim=1)
        hits += (ranks < 10).sum().item()
        ndcg += (
            torch.where(
                ranks < 10,
                1.0 / torch.log2(ranks.float() + 2),
                torch.zeros_like(ranks, dtype=torch.float),
            )
            .sum()
            .item()
        )
        count += len(ranks)
    return hits / count, ndcg / count


def train_vae(name, device, deadline):
    seed_training()
    train, valid, nitems = load_data(name)
    data = Histories(train)
    if not len(data):
        raise ValueError("Empty training dataset: " + name)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        data,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        collate_fn=pack_training_rows,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        generator=generator,
    )
    iterator = iter(loader)
    # Save the architecture flag in every selected MultVAE checkpoint.
    # Dimensions, initialization, objective, optimizer and fusion stay unchanged.
    config = dict(itemnum=nitems, hidden=256, latent=64, dropout=0.5,
                  residual=True)
    model = MultVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    val_batches = validation_batches(train, valid, nitems)
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best = (-1.0, -1.0)
    best_epoch = 0
    update = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        completed = 0
        for _ in range(STEPS):
            try:
                packed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed = next(iterator)
            ids, offsets, owners, target_ids, target_owners = [
                x.to(device, non_blocking=True) for x in packed]
            anneal = min(0.2, update / 10000.0)
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, offsets, owners, anneal,
                         target_ids=target_ids, target_owners=target_owners)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            update += 1
            completed += 1
            loss_sum += float(loss.detach())
            if time.monotonic() >= deadline:
                break
        expired = time.monotonic() >= deadline
        check = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired
        suffix = ""
        if check:
            metric = evaluate(model, val_batches, device)
            suffix = f" val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}"
            if metric > best:
                best = metric
                best_epoch = epoch
                torch.save(
                    {"config": config, "state_dict": model.state_dict()}, checkpoint
                )
        print(
            f"{name} epoch={epoch} loss={loss_sum / max(1, completed):.6f}{suffix}",
            flush=True,
        )
        if expired or epoch - best_epoch >= PATIENCE:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    packed, candidates, histories = val_batches[0]
    reference = expert_scores(model, packed, candidates, histories, device).cpu()
    reloaded = reload_member(name, "_model.pt", MultVAE, device)
    torch.testing.assert_close(
        reference, expert_scores(reloaded, packed, candidates, histories, device).cpu())
    del reloaded
    metric = evaluate(model, val_batches, device)
    del iterator, loader, optimizer
    return model, metric, val_batches


def train_sasrec(name, device, deadline):
    seed_training()
    train, valid, nitems = load_data(name)
    config = dict(
        itemnum=nitems,
        maxlen=int(os.environ.get("SASREC_MAXLEN", "200")),
        hidden=int(os.environ.get("SASREC_HIDDEN", "50")),
        blocks=int(os.environ.get("SASREC_BLOCKS", "2")),
        heads=int(os.environ.get("SASREC_HEADS", "1")),
        dropout=float(os.environ.get("SASREC_DROPOUT", "0.5")),
    )
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(os.environ.get("SASREC_LR", "0.001")),
        betas=(0.9, 0.98))
    batches = validation_batches(train, valid, nitems)
    checkpoint = os.path.join(WORKING, name + "_sasrec.pt")
    best = (-1.0, -1.0)
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        completed = 0
        for _ in range(STEPS):
            seq, pos, neg = [torch.as_tensor(x, device=device) for x in
                             sample_batch(train, nitems, SAS_BATCH, model.maxlen)]
            optimizer.zero_grad(set_to_none=True)
            positive_logits, negative_logits = model(seq, pos, neg)
            # Temperature-one competition, averaged over active transitions.
            loss = F.cross_entropy(
                torch.cat([positive_logits[:, None], negative_logits], dim=1).float(),
                torch.zeros(positive_logits.shape[0], dtype=torch.long,
                            device=device))
            l2 = float(os.environ.get("SASREC_L2_EMB", "0.0"))
            if l2:
                loss = loss + l2 * model.item_emb.weight.norm()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SASRec loss")
            loss.backward()
            optimizer.step()
            completed += 1
            loss_sum += float(loss.detach())
            if time.monotonic() >= deadline:
                break
        expired = time.monotonic() >= deadline
        suffix = ""
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired:
            metric = evaluate(model, batches, device)
            suffix = f" val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}"
            if metric > best:
                best, best_epoch = metric, epoch
                torch.save({"config": config, "state_dict": model.state_dict()},
                           checkpoint)
        print(f"{name} SASRec epoch={epoch} "
              f"loss={loss_sum / max(1, completed):.6f}{suffix}", flush=True)
        if expired or epoch - best_epoch >= PATIENCE:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return model, evaluate(model, batches, device)


def reload_member(name, suffix, constructor, device):
    saved = torch.load(os.path.join(WORKING, name + suffix),
                       map_location="cpu", weights_only=True)
    model = constructor(**saved["config"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def verify_fusion():
    a = torch.tensor([[3.0, 3.0, 1.0, 2.0]])
    b = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    torch.testing.assert_close(descending_ranks(a),
                               torch.tensor([[0.5, 0.5, 3.0, 2.0]]))
    permutation = torch.tensor([2, 0, 3, 1])
    vae_rank, sasrec_rank = descending_ranks(a), descending_ranks(b)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = rank_fusion(a, b, alpha)
        torch.testing.assert_close(scores, fuse_ranks(vae_rank, sasrec_rank, alpha))
        torch.testing.assert_close(
            rank_fusion(a[:, permutation], b[:, permutation], alpha),
            scores[:, permutation])
        assert rank_fusion(a[:, :0], b[:, :0], alpha).shape == (1, 0)
        torch.testing.assert_close(rank_fusion(a[:, :1], b[:, :1], alpha),
                                   torch.zeros(1, 1))
        tied = rank_fusion(torch.ones_like(a), torch.ones_like(b), alpha)
        torch.testing.assert_close(tied, torch.full_like(a, -0.5))
        torch.testing.assert_close(
            torch.argsort(tied, descending=True, stable=True),
            torch.arange(a.shape[-1]).unsqueeze(0))
    torch.testing.assert_close(rank_fusion(a, b), -(vae_rank + sasrec_rank) / 6.0,
                               rtol=0, atol=0)
    torch.testing.assert_close(rank_fusion(a, b, 0.5), rank_fusion(a, b),
                               rtol=0, atol=0)
    for alpha, expert in ((0.0, b), (1.0, a)):
        torch.testing.assert_close(
            torch.argsort(rank_fusion(a, b, alpha), descending=True, stable=True),
            torch.argsort(expert, descending=True, stable=True))


def train_category(name, device, deadline):
    # Give each independent member an explicit share and reserve export checks.
    now = time.monotonic()
    usable = max(0.0, deadline - now - 30.0)
    vae, vae_metric, batches = train_vae(name, device, now + usable * 0.5)
    vae = vae.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sasrec, sasrec_metric = train_sasrec(name, device, deadline - 30.0)
    vae = vae.to(device).eval()
    sasrec.eval()
    packed, candidates, histories = batches[0]
    reference = expert_scores((vae, sasrec), packed, candidates, histories,
                              device).cpu()
    del vae, sasrec
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vae = reload_member(name, "_model.pt", MultVAE, device)
    sasrec = reload_member(name, "_sasrec.pt", SASRec, device)
    reloaded = expert_scores((vae, sasrec), packed, candidates, histories,
                             device).cpu()
    torch.testing.assert_close(reference, reloaded)
    ensemble_metric = evaluate((vae, sasrec), batches, device)
    for label, metric in (("MultVAE", vae_metric), ("SASRec", sasrec_metric),
                          ("equal_rank_ensemble", ensemble_metric)):
        print(f"{name} {label} HR@10={metric[0]:.6f} "
              f"NDCG@10={metric[1]:.6f}", flush=True)
    # Use the actual generated predictor to check the complete inference path.
    spec = importlib.util.spec_from_file_location(
        "export_verification", os.path.join(WORKING, "predict.py"))
    exported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exported)
    local = expert_scores((vae, sasrec), pack_rows([
        np.asarray(sorted(set(histories[0])), dtype=np.int64)]),
        candidates[:1], histories[:1], device)[0].cpu()
    del vae, sasrec
    if device.type == "cuda":
        torch.cuda.empty_cache()
    actual = exported.predict(0, histories[0], candidates[0].tolist(),
                              dataset_name=name)
    torch.testing.assert_close(local, torch.tensor(actual))
    assert exported.predict(0, [], [], dataset_name=name) == []
    empty_scores = exported.predict(0, [], candidates[0].tolist(),
                                    dataset_name=name)
    assert np.isfinite(empty_scores).all()
    exported._models.clear()
    del exported
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ensemble_metric[0]


def verify_masked_objective():
    # Synthetic checks preserve all training RNG streams and use no split data.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            data = Histories({1: [3, 1, 3, 2], 2: [4, 4],
                              3: list(range(1, 12))})
            np.testing.assert_array_equal(data.rows[0], [1, 2, 3])
            rows = [data[i] for i in range(len(data))]
            for original, (visible, target) in zip(data.rows, rows):
                expected = (1 if len(original) == 1 else
                            min(len(original) - 1,
                                max(1, int(math.floor(0.2 * len(original))))))
                assert len(target) == expected
                assert len(np.unique(target)) == len(target)
                if len(original) == 1:
                    np.testing.assert_array_equal(visible, original)
                    np.testing.assert_array_equal(target, original)
                else:
                    assert not np.intersect1d(visible, target).size
                    np.testing.assert_array_equal(
                        np.sort(np.concatenate((visible, target))), original)
            ids, offsets, owners, target_ids, target_owners = pack_training_rows(rows)
            for index, (_, target) in enumerate(rows):
                np.testing.assert_array_equal(
                    target_ids[target_owners == index].numpy(), target)
            model = MultVAE(11, hidden=8, latent=3, dropout=0.0, residual=True)
            model.eval()
            h, mu, logvar = model.representation(ids, offsets)
            logits = model.decoder(h).float()
            logits[:, 0] = -1e9
            logprobs = F.log_softmax(logits, dim=-1)
            direct_nll = -logprobs[target_owners, target_ids].sum() / len(rows)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
            actual_nll = model(ids, offsets, owners, 0.0, target_ids, target_owners)
            actual = model(ids, offsets, owners, 0.17, target_ids, target_owners)
            torch.testing.assert_close(actual_nll, direct_nll)
            torch.testing.assert_close(actual, direct_nll + 0.17 * kl)
            torch.testing.assert_close(actual - actual_nll, 0.17 * kl)
            original = model(ids, offsets, owners, 0.17)
            torch.testing.assert_close(
                original, -logprobs[owners, ids].sum() / len(rows) + 0.17 * kl)
            actual.backward()
            for parameter in model.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all().item()
            assert model.encoder_items.weight.grad.abs().sum().item() > 0
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


FUSION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def stage_fusion_checkpoints():
    # Stage only the unchanged SASRec experts and the previously selected global
    # fusion configuration. Newly trained VAE files are never staging targets.
    import hashlib
    import shutil

    source_dir = os.path.abspath(os.environ.get(
        "FUSION_CHECKPOINT_DIR",
        "/home/liuzhishan/lhr/rec-harness/artifacts/amazon-aligned-20261022/"
        "mlevolve/trials/456cdfd6ee544e149f1f1d9bdfbc27de/working"))
    fusion_source = os.path.join(source_dir, "fusion.json")
    with open(fusion_source) as handle:
        fixed = json.load(handle)
    alpha = fixed["alpha"]
    if (alpha not in FUSION_ALPHAS or fixed["vae_weight"] != alpha or
            fixed["sasrec_weight"] != 1.0 - alpha):
        raise ValueError("Invalid previously selected fusion configuration")
    pending = []
    for name in DATASETS:
        for suffix in ("_sasrec.pt",):
            source = os.path.join(source_dir, name + suffix)
            destination = os.path.join(WORKING, name + suffix)
            if not os.path.isfile(source):
                raise FileNotFoundError(
                    "Masked VAE training requires unchanged SASRec checkpoint: "
                    + source)
            pending.append((name, suffix, source, destination))
    provenance = {}
    for name, suffix, source, destination in pending:
        if os.path.realpath(source) != os.path.realpath(destination):
            shutil.copy2(source, destination)
        def digest(path):
            value = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    value.update(chunk)
            return value.hexdigest()
        source_hash, staged_hash = digest(source), digest(destination)
        if source_hash != staged_hash:
            raise RuntimeError("Checkpoint staging changed bytes: " + source)
        saved = torch.load(destination, map_location="cpu", weights_only=True)
        if suffix == "_model.pt" and saved["config"].get("residual") is not True:
            raise ValueError("Required residual MultVAE checkpoint: " + source)
        provenance.setdefault(name, {})[suffix] = {
            "source": source, "staged": os.path.basename(destination),
            "sha256": staged_hash, "config": saved["config"],
            "selection": "existing independently selected expert; unchanged",
        }
    fusion_destination = os.path.join(WORKING, "fusion.json")
    if os.path.realpath(fusion_source) != os.path.realpath(fusion_destination):
        shutil.copy2(fusion_source, fusion_destination)
    if digest(fusion_source) != digest(fusion_destination):
        raise RuntimeError("Fixed fusion configuration changed during staging")
    provenance["fixed_fusion"] = {
        "source": fusion_source, "sha256": digest(fusion_destination),
        "alpha": alpha, "selection": "unchanged previously selected global alpha"}
    return provenance


@torch.inference_mode()
def cache_validation_ranks(name, device):
    train, valid, nitems = load_data(name)
    batches = validation_batches(train, valid, nitems)
    vae = reload_member(name, "_model.pt", MultVAE, device)
    sasrec = reload_member(name, "_sasrec.pt", SASRec, device)
    if vae.itemnum != nitems or sasrec.itemnum != nitems:
        raise ValueError("Checkpoint catalog differs from supplied data: " + name)
    cached = []
    for packed, candidates, histories in batches:
        vae_scores = expert_scores(vae, packed, candidates, histories, device)
        sasrec_scores = expert_scores(sasrec, packed, candidates, histories, device)
        cached.append((descending_ranks(vae_scores).cpu(),
                       descending_ranks(sasrec_scores).cpu()))
    packed, candidates, histories = batches[0]
    example = (list(histories[0]), candidates[0].tolist())
    del vae, sasrec
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached, example


def evaluate_cached_ranks(cached, alpha):
    hits = ndcg = count = 0
    for vae_rank, sasrec_rank in cached:
        scores = fuse_ranks(vae_rank, sasrec_rank, alpha)
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        ranks = (order == 0).long().argmax(dim=1)
        hits += int((ranks < 10).sum())
        ndcg += float(torch.where(
            ranks < 10, 1.0 / torch.log2(ranks.float() + 2),
            torch.zeros_like(ranks, dtype=torch.float)).sum())
        count += len(ranks)
    return {"HR@10": hits / count, "NDCG@10": ndcg / count,
            "n_users": count}


@torch.inference_mode()
def verify_exported_fusion(examples, device, deadline):
    with open(os.path.join(WORKING, "fusion.json")) as handle:
        metadata = json.load(handle)
    for name in DATASETS:
        if time.monotonic() >= deadline:
            raise RuntimeError("Fusion diagnostic budget exhausted")
        history, candidates = examples[name]
        vae = reload_member(name, "_model.pt", MultVAE, device)
        sasrec = reload_member(name, "_sasrec.pt", SASRec, device)
        candidate_tensor = torch.tensor([candidates], dtype=torch.long)
        direct = []
        for row_history in (history, []):
            packed = pack_rows([np.asarray(sorted(set(row_history)), dtype=np.int64)])
            direct.append((
                expert_scores(vae, packed, candidate_tensor, [row_history], device).cpu(),
                expert_scores(sasrec, packed, candidate_tensor, [row_history], device).cpu()))
        del vae, sasrec
        if device.type == "cuda":
            torch.cuda.empty_cache()
        spec = importlib.util.spec_from_file_location(
            "export_verification_" + name, os.path.join(WORKING, "predict.py"))
        exported = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exported)
        selected = exported.predict(0, history, candidates, dataset_name=name)
        assert exported._fusion == metadata
        torch.testing.assert_close(torch.tensor(selected),
                                   rank_fusion(*direct[0], metadata["alpha"])[0])
        permutation = list(reversed(range(len(candidates))))
        for alpha in (metadata["alpha"],):
            exported._fusion = dict(metadata, alpha=alpha,
                                    vae_weight=alpha, sasrec_weight=1.0 - alpha)
            for row_history, scores in zip((history, []), direct):
                expected = rank_fusion(*scores, alpha)[0]
                actual = exported.predict(0, row_history, candidates, dataset_name=name)
                torch.testing.assert_close(torch.tensor(actual), expected)
                actual = exported.predict(
                    0, row_history, [candidates[i] for i in permutation],
                    dataset_name=name)
                torch.testing.assert_close(torch.tensor(actual), expected[permutation])
            assert exported.predict(0, [], [], dataset_name=name) == []
        # Force metadata reload after checking the unchanged selected weight.
        exported._fusion = None
        reloaded = exported.predict(0, history, candidates, dataset_name=name)
        assert exported._fusion == metadata
        torch.testing.assert_close(torch.tensor(reloaded), torch.tensor(selected))
        if name == DATASETS[0]:
            torch.testing.assert_close(
                torch.tensor(exported.predict(0, history, candidates)),
                torch.tensor(selected))
        exported._models.clear()
        del exported
        if device.type == "cuda":
            torch.cuda.empty_cache()


def fusion_only_evaluation(device, deadline):
    provenance = stage_fusion_checkpoints()
    results = [{"alpha": alpha, "vae_weight": alpha,
                "sasrec_weight": 1.0 - alpha, "per_dataset": {}}
               for alpha in FUSION_ALPHAS]
    examples = {}
    for name in DATASETS:
        if time.monotonic() >= deadline:
            raise RuntimeError("Fusion evaluation budget exhausted; no retraining")
        cached, examples[name] = cache_validation_ranks(name, device)
        for result in results:
            metric = evaluate_cached_ranks(cached, result["alpha"])
            result["per_dataset"][name] = metric
            print(f"{name} alpha={result['alpha']:.2f} "
                  f"HR@10={metric['HR@10']:.6f} "
                  f"NDCG@10={metric['NDCG@10']:.6f}", flush=True)
        del cached
    for result in results:
        result["macro"] = {
            key: float(np.mean([result["per_dataset"][name][key]
                                for name in DATASETS]))
            for key in ("HR@10", "NDCG@10")}
    selected = min(results, key=lambda result: (
        -result["macro"]["HR@10"], abs(result["alpha"] - 0.5), result["alpha"]))
    baseline = next(result for result in results if result["alpha"] == 0.5)
    delta = selected["macro"]["HR@10"] - baseline["macro"]["HR@10"]
    assert delta >= 0.0
    metadata = {
        "alpha": selected["alpha"], "vae_weight": selected["alpha"],
        "sasrec_weight": 1.0 - selected["alpha"],
        "orientation": "alpha * VAE average rank + (1-alpha) * SASRec average rank",
        "selection": "macro HR@10 descending, distance to 0.5 ascending, alpha ascending",
        "results": results, "checkpoint_provenance": provenance,
        "preprocessing_seed": 0, "training_seed": 42, "evaluation_seed": 42,
        "same_cache_delta_vs_alpha_0.5": delta,
        "host_improvement": "unproven; host validation is evaluated independently",
        "NDCG": "diagnostic only",
    }
    with open(os.path.join(WORKING, "fusion.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    write_predict_script()
    verify_exported_fusion(examples, device, deadline)
    print(f"Selected global VAE weight={selected['alpha']:.2f}; "
          f"same-cache HR@10 delta versus alpha=0.5={delta:+.6f}; "
          "host improvement remains unproven", flush=True)
    return selected["macro"]["HR@10"]


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import json
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_module = None
_device = None
_fusion = None


def _load(dataset_name):
    global _module, _device, _fusion
    if _fusion is None:
        with open(os.path.join(_HERE, 'fusion.json')) as handle:
            metadata = json.load(handle)
        alpha = metadata['alpha']
        if alpha not in (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError('Invalid global fusion alpha')
        if (metadata['vae_weight'] != alpha or
                metadata['sasrec_weight'] != 1.0 - alpha):
            raise ValueError('Inconsistent fusion weight orientation')
        _fusion = metadata
    if _module is None:
        spec = importlib.util.spec_from_file_location(
            'trained_multvae_module', os.path.join(_HERE, '..', 'train.py'))
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    if dataset_name not in _models:
        saved = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                           map_location='cpu', weights_only=True)
        # Masking is training-only. Load the saved residual architecture and use
        # complete binary histories for both validation and held-out inference.
        config = dict(saved['config'])
        if config.get('residual') is not True:
            raise ValueError('Expected revised residual MultVAE checkpoint')
        model = _module.MultVAE(**config)
        model.load_state_dict(saved['state_dict'])
        sequential = torch.load(os.path.join(_HERE, dataset_name + '_sasrec.pt'),
                                map_location='cpu', weights_only=True)
        sasrec = _module.SASRec(**sequential['config'])
        sasrec.load_state_dict(sequential['state_dict'])
        _models[dataset_name] = (model.to(_device).eval(),
                                 sasrec.to(_device).eval())
    return _models[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    if len(candidates) == 0:
        return []
    model, sasrec = _load(ds)
    # Identical binary, untruncated history representation to validation.
    row = np.asarray(sorted(set(int(i) for i in history)), dtype=np.int64)
    if row.size and (row.min() < 1 or row.max() > model.itemnum):
        raise ValueError('History item outside the trained catalog')
    ids, offsets, _ = _module.pack_rows([row])
    candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=_device)
    if candidate_tensor.min() < 1 or candidate_tensor.max() > model.itemnum:
        raise ValueError('Candidate item outside the trained catalog')
    vae_scores = model.predict(ids.to(_device), offsets.to(_device), candidate_tensor)
    sequence = _module.chronological_rows([history], sasrec.maxlen, _device)
    sequential_scores = sasrec.predict(sequence, candidate_tensor)
    scores = _module.rank_fusion(vae_scores, sequential_scores, _fusion['alpha'])
    return scores[0].cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as handle:
        handle.write(script)


def masked_training_evaluation(device, deadline):
    provenance = stage_fusion_checkpoints()
    with open(os.path.join(WORKING, "fusion.json")) as handle:
        fixed = json.load(handle)
    alpha = fixed["alpha"]
    source_dir = os.path.dirname(provenance["fixed_fusion"]["source"])
    if os.path.realpath(source_dir) == os.path.realpath(WORKING):
        raise ValueError("Reference checkpoints must reside outside this trial")
    write_predict_script()
    results, unchanged, examples = {}, {}, {}
    latest = {"Movies_and_TV": 0.645, "Industrial_and_Scientific": 0.625,
              "Electronics": 0.470, "CDs_and_Vinyl": 0.685}
    # Reserve export time before dividing the remaining allowance equally among
    # categories. Unused early-stopping time flows to the remaining categories.
    available = deadline - time.monotonic()
    diagnostic_reserve = min(300.0, available * 0.15)
    training_end = deadline - diagnostic_reserve
    for index, name in enumerate(DATASETS):
        now = time.monotonic()
        categories_left = len(DATASETS) - index
        share = (training_end - now) / categories_left
        category_reserve = min(60.0, share * 0.15)
        if share <= 0.0:
            raise RuntimeError("No training budget remains for " + name)
        category_deadline = now + share
        vae, vae_metric, batches = train_vae(
            name, device, category_deadline - category_reserve)
        vae = vae.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sasrec = reload_member(name, "_sasrec.pt", SASRec, device)
        reference_path = os.path.join(source_dir, name + "_model.pt")
        saved = torch.load(reference_path, map_location="cpu", weights_only=True)
        if saved["config"].get("residual") is not True:
            raise ValueError("Expected unchanged residual reference VAE")
        reference_vae = MultVAE(**saved["config"])
        reference_vae.load_state_dict(saved["state_dict"])
        reference_vae = reference_vae.to(device).eval()
        sasrec_ranks, reference_ranks = [], []
        for packed, candidates, histories in batches:
            sr = descending_ranks(expert_scores(
                sasrec, packed, candidates, histories, device)).cpu()
            vr = descending_ranks(expert_scores(
                reference_vae, packed, candidates, histories, device)).cpu()
            sasrec_ranks.append(sr)
            reference_ranks.append((vr, sr))
        unchanged[name] = evaluate_cached_ranks(reference_ranks, alpha)
        del reference_vae, saved, reference_ranks
        if device.type == "cuda":
            torch.cuda.empty_cache()
        vae = vae.to(device).eval()
        revised_ranks = []
        for (packed, candidates, histories), sr in zip(batches, sasrec_ranks):
            vr = descending_ranks(expert_scores(
                vae, packed, candidates, histories, device)).cpu()
            revised_ranks.append((vr, sr))
        results[name] = evaluate_cached_ranks(revised_ranks, alpha)
        packed, candidates, histories = batches[0]
        examples[name] = (list(histories[0]), candidates[0].tolist())
        del vae, sasrec
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Check every pinned validation user through the generated predictor,
        # using complete unmasked histories and the exact same candidates.
        spec = importlib.util.spec_from_file_location(
            "masked_export_" + name, os.path.join(WORKING, "predict.py"))
        exported = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exported)
        for (_, candidates, histories), (vr, sr) in zip(batches, revised_ranks):
            if time.monotonic() >= category_deadline:
                raise RuntimeError("Category training/export allowance exhausted")
            actual = torch.tensor([
                exported.predict(0, history, row.tolist(), dataset_name=name)
                for history, row in zip(histories, candidates)])
            torch.testing.assert_close(actual, fuse_ranks(vr, sr, alpha))
        exported._models.clear()
        del exported, revised_ranks, sasrec_ranks, batches
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{name} masked_VAE HR@10={vae_metric[0]:.6f} "
              f"NDCG@10={vae_metric[1]:.6f}; fixed ensemble="
              f"{results[name]['HR@10']:.6f}; unchanged ensemble="
              f"{unchanged[name]['HR@10']:.6f}; latest reported="
              f"{latest[name]:.6f}; delta_vs_unchanged="
              f"{results[name]['HR@10'] - unchanged[name]['HR@10']:+.6f}",
              flush=True)
    verify_exported_fusion(examples, device, deadline)
    macro = {key: float(np.mean([results[name][key] for name in DATASETS]))
             for key in ("HR@10", "NDCG@10")}
    reference_macro = {
        key: float(np.mean([unchanged[name][key] for name in DATASETS]))
        for key in ("HR@10", "NDCG@10")}
    report = {
        "experiment": "20 percent binary masked-item denoising",
        "per_dataset": results, "macro": macro,
        "unchanged_ensemble": {"per_dataset": unchanged, "macro": reference_macro},
        "fixed_alpha": alpha, "checkpoint_provenance": provenance,
        "preprocessing_seed": 0, "training_seed": 42, "evaluation_seed": 42,
        "validation_users_per_category": 200, "unique_negatives_per_user": 99,
        "latest_reported_macro_HR@10": 0.6062,
        "best_reported_macro_HR@10": 0.6250,
        "delta_vs_unchanged": macro["HR@10"] - reference_macro["HR@10"],
        "delta_vs_latest": macro["HR@10"] - 0.6062,
        "delta_vs_reported_best": macro["HR@10"] - 0.6250,
        "checkpoint_selection": "VAE validation HR@10, then NDCG@10",
        "objective_note": "Fewer reconstruction targets change reconstruction/KL balance; KL and annealing unchanged",
        "held_out_inference": "host only; improvement unproven",
    }
    with open(os.path.join(WORKING, "masked_diagnostics.json"), "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"Measured macro fixed-ensemble HR@10={macro['HR@10']:.6f}; "
          f"unchanged={reference_macro['HR@10']:.6f}; "
          f"delta versus latest 0.6062={report['delta_vs_latest']:+.6f}; "
          f"delta versus reported best 0.6250="
          f"{report['delta_vs_reported_best']:+.6f}", flush=True)
    return macro["HR@10"]


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(WORKING, exist_ok=True)
    # Sequential single-GPU training, validation, staging and export checks all
    # consume the remaining shared search allowance; no held-out data is read.
    started = time.monotonic()
    known_remaining = (43200.0 - 1757.786672219634 - 104.27287103235722
                       - 903.5522981882095 - 944.6417525559664
                       - 2007.1221108585596 - 13.63338640332222)
    remaining = min(known_remaining, float(os.environ.get(
        "REMAINING_GPU_SECONDS", str(known_remaining))))
    if not math.isfinite(remaining) or remaining <= 0.0:
        raise RuntimeError("No remaining masked-training budget")
    deadline = started + remaining
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    threads = torch.get_num_threads()
    try:
        # Synthetic checks preserve RNG; each category also resets training to 42.
        devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            verify_fusion()
            verify_masked_objective()
            score = masked_training_evaluation(device, deadline)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_num_threads(threads)
    if time.monotonic() >= deadline:
        raise RuntimeError("Masked training and diagnostics exceeded budget")
    print(f"Measured fixed-ensemble macro validation HR@10={score:.6f}")
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
