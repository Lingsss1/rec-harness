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
        return self.rows[index]


def pack_rows(rows):
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    ids = np.concatenate(rows) if offsets[-1] else np.empty(0, dtype=np.int64)
    owners = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
    return (torch.from_numpy(ids), torch.from_numpy(offsets), torch.from_numpy(owners))


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

    def forward(self, ids, offsets, owners, anneal):
        h, mu, logvar = self.representation(ids, offsets)
        logits = self.decoder(h).float()
        logits[:, 0] = -1e9
        logprobs = F.log_softmax(logits, dim=-1)
        # Standard multinomial reconstruction: every observed training item
        # contributes once. Validation items never appear in these targets.
        nll = -logprobs[owners, ids].sum() / h.shape[0]
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


def rank_fusion(vae_scores, sasrec_scores):
    if vae_scores.shape != sasrec_scores.shape:
        raise ValueError("Expert candidate shapes differ")
    denominator = max(vae_scores.shape[-1] - 1, 1)
    return -(descending_ranks(vae_scores) + descending_ranks(sasrec_scores)) / (
        2.0 * denominator)


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
        collate_fn=pack_rows,
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
            ids, offsets, owners = [x.to(device, non_blocking=True) for x in packed]
            anneal = min(0.2, update / 10000.0)
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, offsets, owners, anneal)
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
    torch.testing.assert_close(rank_fusion(a[:, permutation], b[:, permutation]),
                               rank_fusion(a, b)[:, permutation])
    assert rank_fusion(a[:, :0], b[:, :0]).shape == (1, 0)
    torch.testing.assert_close(rank_fusion(a[:, :1], b[:, :1]),
                               torch.zeros(1, 1))


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


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_module = None
_device = None


def _load(dataset_name):
    global _module, _device
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
        # Reconstruct the saved residual architecture; older checkpoints default
        # to the original architecture. History preprocessing remains identical.
        config = dict(saved['config'])
        config.setdefault('residual', False)
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
    scores = _module.rank_fusion(vae_scores, sequential_scores)
    return scores[0].cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as handle:
        handle.write(script)


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
    write_predict_script()
    verify_fusion()
    verify_sasrec_objective()
    # Known prior allocations are already spent. The host can supply a smaller
    # remaining allowance when additional trials have consumed shared budget.
    known_remaining = (43200.0 - 1757.786672219634 - 104.27287103235722
                       - 903.5522981882095 - 944.6417525559664)
    remaining = min(known_remaining, float(os.environ.get(
        "REMAINING_GPU_SECONDS", str(known_remaining))))
    if remaining <= 180.0:
        raise RuntimeError("Insufficient remaining budget for four categories")
    end = time.monotonic() + min(8 * 3600.0, remaining - 120.0)
    scores = []
    for index, name in enumerate(DATASETS):
        remaining_categories = len(DATASETS) - index
        category_deadline = time.monotonic() + max(
            1.0, (end - time.monotonic()) / remaining_categories
        )
        scores.append(train_category(name, device, category_deadline))
    score = float(np.mean(scores))
    # Category metrics and reload/export agreement are checked in train_category.
    # The proposed objective gain remains unverified until this run is evaluated.
    # Per-category member and ensemble metrics are printed in train_category.
    # The predictor writer imports SASRec from train.py; inference is unchanged.
    print(f"Measured macro ensemble HR@10={score:.6f}; "
          f"delta versus supplied residual ensemble 0.6250={score - 0.6250:+.6f}")
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
