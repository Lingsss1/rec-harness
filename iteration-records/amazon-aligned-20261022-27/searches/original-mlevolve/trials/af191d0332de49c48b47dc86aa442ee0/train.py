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


def transform_history(items):
    # Restore the successful uniform, binary-history reconstruction input.
    ids = np.unique(np.asarray(list(items), dtype=np.int64))
    weights = np.full(ids.size, 1.0 / math.sqrt(max(1, ids.size)), dtype=np.float32)
    return ids, weights


class Histories(Dataset):
    def __init__(self, train):
        self.rows = [list(items) for items in train.values() if items]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return transform_history(self.rows[index])


def pack_rows(rows):
    lengths = np.asarray([len(row[0]) for row in rows], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    ids = np.concatenate([row[0] for row in rows]) if offsets[-1] else np.empty(0, dtype=np.int64)
    weights = np.concatenate([row[1] for row in rows]) if offsets[-1] else np.empty(0, dtype=np.float32)
    owners = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
    # Unique IDs and owners define binary targets; weights normalize inputs only.
    return (
        torch.from_numpy(ids),
        torch.from_numpy(offsets),
        torch.from_numpy(owners),
        torch.from_numpy(weights),
    )


def pack_training_rows(rows):
    return pack_rows(rows)


def prepare_sequences(histories, maxlen=50):
    result = torch.zeros((len(histories), maxlen), dtype=torch.long)
    for index, history in enumerate(histories):
        tail = list(history)[-maxlen:]
        if tail:
            result[index, -len(tail):] = torch.tensor(tail, dtype=torch.long)
    return result


class SequentialHistories(Dataset):
    # Separate chronological sampler: sparse reconstruction packing is unchanged.
    def __init__(self, train, nitems, maxlen):
        self.rows = [list(items) for items in train.values() if len(items) > 1]
        self.seen = [set(items) for items in self.rows]
        self.nitems = nitems
        self.maxlen = maxlen
        if not self.rows or any(len(seen) >= nitems for seen in self.seen):
            raise ValueError("SASRec needs eligible users and unseen negatives")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        items, seen = self.rows[index], self.seen[index]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros_like(seq)
        # Independent uniform draws with replacement; padded rows stay zero.
        neg = np.zeros((self.maxlen, 16), dtype=np.int64)
        next_item = items[-1]
        position = self.maxlen - 1
        for item in reversed(items[:-1]):
            seq[position] = item
            pos[position] = next_item
            for competitor in range(16):
                sampled = random.randint(1, self.nitems)
                while sampled in seen:
                    sampled = random.randint(1, self.nitems)
                neg[position, competitor] = sampled
            next_item = item
            position -= 1
            if position < 0:
                break
        return seq, pos, neg


def pack_sequence_rows(rows):
    return tuple(torch.from_numpy(np.stack(part)) for part in zip(*rows))


def seed_training_worker(worker_id):
    # DataLoader derives each worker's Torch seed from its seed-42 generator.
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class MultVAE(nn.Module):
    def __init__(self, itemnum, hidden=256, latent=64, dropout=0.5):
        super().__init__()
        self.itemnum = itemnum
        self.dropout = dropout
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

    def encode(self, ids, offsets, input_weights=None):
        if input_weights is None:
            lengths = offsets[1:] - offsets[:-1]
            weights = torch.repeat_interleave(lengths.clamp_min(1).float().rsqrt(), lengths)
        else:
            weights = input_weights
        # Explicit uniform weights agree with the length-based fallback.
        # Apply corruption after normalization without renormalizing.
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        h = torch.tanh(
            self.encoder_items(ids, offsets, per_sample_weights=weights)
            + self.encoder_bias
        )
        mu, logvar = self.posterior(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-12.0, 8.0)

    def representation(self, ids, offsets, input_weights=None):
        mu, logvar = self.encode(ids, offsets, input_weights)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return torch.tanh(self.decoder_hidden(z)), mu, logvar

    def forward(self, ids, offsets, owners, anneal, input_weights=None, targets=None):
        if targets is not None:
            raise ValueError("MultVAE uses binary full-history reconstruction")
        h, mu, logvar = self.representation(ids, offsets, input_weights)
        logprobs = F.log_softmax(self.decoder(h).float(), dim=-1)
        # Each unique observed item contributes once, without target weighting.
        nll = -logprobs[owners, ids].sum() / h.shape[0]
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return nll + anneal * kl

    def predict(self, ids, offsets, candidates, input_weights=None):
        h, _, _ = self.representation(ids, offsets, input_weights)
        weights = F.embedding(candidates, self.decoder.weight)
        biases = self.decoder.bias[candidates]
        return (weights * h.unsqueeze(1)).sum(-1) + biases


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(1, 2))
        outputs = F.relu(self.dropout1(outputs))
        outputs = self.dropout2(self.conv2(outputs)).transpose(1, 2)
        return outputs + inputs


class SASRec(nn.Module):
    # Standard SASRec architecture with positive-plus-16-negative cross-entropy.
    def __init__(self, itemnum, hidden=50, maxlen=50, blocks=2, heads=1, dropout=0.5):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.hidden = hidden
        self.item_emb = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.emb_dropout = nn.Dropout(dropout)
        self.attention_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=dropout)
            for _ in range(blocks)
        ])
        self.forward_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks)
        ])
        self.forward_layers = nn.ModuleList([
            PointWiseFeedForward(hidden, dropout) for _ in range(blocks)
        ])
        self.last_layernorm = nn.LayerNorm(hidden, eps=1e-8)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, log_seqs):
        padding = log_seqs.eq(0)
        seqs = self.item_emb(log_seqs) * math.sqrt(self.hidden)
        positions = torch.arange(log_seqs.shape[1], device=log_seqs.device)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions).unsqueeze(0))
        seqs = seqs.masked_fill(padding.unsqueeze(-1), 0.0)
        causal = torch.ones(
            log_seqs.shape[1], log_seqs.shape[1],
            dtype=torch.bool, device=log_seqs.device,
        ).triu(1)
        for norm, attention, forward_norm, forward in zip(
            self.attention_layernorms, self.attention_layers,
            self.forward_layernorms, self.forward_layers,
        ):
            seqs = seqs.transpose(0, 1)
            queries = norm(seqs)
            attended, _ = attention(
                queries, seqs, seqs, attn_mask=causal, need_weights=False
            )
            seqs = (queries + attended).transpose(0, 1)
            seqs = forward(forward_norm(seqs))
            seqs = seqs.masked_fill(padding.unsqueeze(-1), 0.0)
        return self.last_layernorm(seqs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """Return active positive (N_active,) and negative (N_active, 16) logits."""
        features = self.log2feats(log_seqs)
        active = pos_seqs.ne(0)
        features = features[active]
        positive = (features * self.item_emb(pos_seqs[active])).sum(-1)
        # Gather before embedding to avoid allocating padded negative embeddings.
        negative = (
            features.unsqueeze(1) * self.item_emb(neg_seqs[active])
        ).sum(-1)
        return positive, negative

    def predict(self, user_ids, log_seqs, item_indices):
        features = self.log2feats(log_seqs)[:, -1]
        return (self.item_emb(item_indices) * features.unsqueeze(1)).sum(-1)


def descending_ranks(scores):
    # Stable sorting gives the original candidate order precedence on ties.
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(scores.shape[-1], device=scores.device)
    ranks.scatter_(-1, order, positions.expand_as(order))
    return ranks


def rank_fusion(vae_scores, sasrec_scores):
    r_vae = descending_ranks(vae_scores).float() + 1.0
    r_sasrec = descending_ranks(sasrec_scores).float() + 1.0
    return 0.75 / (10.0 + r_vae) + 0.25 / (10.0 + r_sasrec)


def score_members(vae, sasrec, packed, sequences, candidates):
    ids, offsets, _, input_weights = packed
    vae_scores = vae.predict(ids, offsets, candidates, input_weights)
    sasrec_scores = sasrec.predict(None, sequences, candidates)
    return vae_scores, sasrec_scores, rank_fusion(vae_scores, sasrec_scores)


class Ensemble(nn.Module):
    def __init__(self, vae, sasrec, dataset_name=None):
        super().__init__()
        self.vae = vae
        self.sasrec = sasrec
        self.itemnum = vae.itemnum
        self.dataset_name = dataset_name

    def score(self, packed, sequences, candidates):
        return score_members(self.vae, self.sasrec, packed, sequences, candidates)[2]

    @torch.inference_mode()
    def predict(self, user_id, history, candidates, *, dataset_name=None):
        if dataset_name is not None and dataset_name != self.dataset_name:
            raise ValueError("Dataset does not match the loaded ensemble")
        candidates = list(candidates)
        if not candidates:
            return []
        history = [int(item) for item in history]
        if any(item < 1 or item > self.itemnum for item in history + candidates):
            raise ValueError("Item outside the trained catalog")
        self.eval()
        device = next(self.parameters()).device
        packed = tuple(t.to(device) for t in pack_rows([transform_history(history)]))
        sequences = prepare_sequences([history], self.sasrec.maxlen).to(device)
        candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=device)
        return self.score(packed, sequences, candidate_tensor)[0].cpu().tolist()


def load_ensemble(directory, name, device):
    # Reload actual saved configurations and weights for both local and host use.
    vae_saved = torch.load(
        os.path.join(directory, name + "_model.pt"),
        map_location="cpu", weights_only=True,
    )
    sas_saved = torch.load(
        os.path.join(directory, name + "_sasrec.pt"),
        map_location="cpu", weights_only=True,
    )
    # Constructor initialization must not perturb subsequent training RNG streams.
    with torch.random.fork_rng(devices=[]):
        vae = MultVAE(**vae_saved["config"])
        sasrec = SASRec(**sas_saved["config"])
    vae.load_state_dict(vae_saved["state_dict"])
    sasrec.load_state_dict(sas_saved["state_dict"])
    return Ensemble(vae, sasrec, name).to(device).eval()


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
                transform_history(train[u]),
                [target] + list(negatives),
                train[u],
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
                prepare_sequences([x[2] for x in part]),
            )
        )
    return batches


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    hits = ndcg = count = 0
    for packed, candidates, sequences in batches:
        packed = tuple(x.to(device) for x in packed)
        candidates = candidates.to(device)
        sequences = sequences.to(device)
        if isinstance(model, Ensemble):
            scores = model.score(packed, sequences, candidates)
        elif isinstance(model, SASRec):
            scores = model.predict(None, sequences, candidates)
        else:
            ids, offsets, _, input_weights = packed
            scores = model.predict(ids, offsets, candidates, input_weights)
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


def train_sasrec(name, train, nitems, val_batches, device, deadline):
    # Train independently; no shared parameters, targets, or fitted fusion weights.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    config = dict(itemnum=nitems, hidden=50, maxlen=50, blocks=2, heads=1, dropout=0.5)
    model = SASRec(**config).to(device)
    data = SequentialHistories(train, nitems, config["maxlen"])
    generator = torch.Generator().manual_seed(42)
    sampler = torch.utils.data.RandomSampler(
        data, replacement=True, num_samples=STEPS * 128, generator=generator
    )
    loader = DataLoader(
        data, batch_size=128, sampler=sampler, num_workers=2,
        collate_fn=pack_sequence_rows, worker_init_fn=seed_training_worker,
        generator=generator, persistent_workers=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    checkpoint = os.path.join(WORKING, name + "_sasrec.pt")
    best, best_epoch = (-1.0, -1.0), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = completed = 0
        for seq, pos, neg in loader:
            seq, pos, neg = (x.to(device, non_blocking=True) for x in (seq, pos, neg))
            positive, negative = model(None, seq, pos, neg)
            # Each active position contributes equally to the 17-way objective.
            loss = F.cross_entropy(
                torch.cat([positive.unsqueeze(-1), negative], dim=-1).float(),
                torch.zeros(positive.shape[0], dtype=torch.long, device=device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SASRec loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            completed += 1
            loss_sum += float(loss.detach())
            if time.monotonic() >= deadline:
                break
        expired = time.monotonic() >= deadline
        suffix = ""
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired:
            metric = evaluate(model, val_batches, device)
            suffix = f" val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}"
            if metric > best:
                best, best_epoch = metric, epoch
                torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)
        print(
            f"{name} SASRec epoch={epoch} loss={loss_sum / max(1, completed):.6f}{suffix}",
            flush=True,
        )
        if expired or epoch - best_epoch >= PATIENCE:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    selected = evaluate(model, val_batches, device)
    np.testing.assert_allclose(selected, best, rtol=0, atol=1e-12)
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return selected


def train_category(name, device, deadline):
    # Divide each category's allowance between its two independent members.
    # The caller reserves equal allowances for every remaining category.
    member_deadline = time.monotonic() + max(0.0, (deadline - time.monotonic()) / 2)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
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
        worker_init_fn=seed_training_worker,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        generator=generator,
    )
    iterator = iter(loader)
    config = dict(itemnum=nitems, hidden=256, latent=64, dropout=0.5)
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
            ids, offsets, owners, input_weights = [x.to(device, non_blocking=True) for x in packed]
            anneal = min(0.2, update / 10000.0)
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, offsets, owners, anneal, input_weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            update += 1
            completed += 1
            loss_sum += float(loss.detach())
            if time.monotonic() >= member_deadline:
                break
        expired = time.monotonic() >= member_deadline
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
    vae_metric = evaluate(model, val_batches, device)
    np.testing.assert_allclose(vae_metric, best, rtol=0, atol=1e-12)
    del iterator, loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sas_metric = train_sasrec(name, train, nitems, val_batches, device, deadline)
    ensemble = load_ensemble(WORKING, name, device)
    # Confirm both members survive reconstruction from their saved configurations.
    np.testing.assert_allclose(
        evaluate(ensemble.vae, val_batches, device), vae_metric, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        evaluate(ensemble.sasrec, val_batches, device), sas_metric, rtol=0, atol=1e-12
    )
    ensemble_metric = evaluate(ensemble, val_batches, device)
    check_predictor_agreement(name, ensemble, train, valid, val_batches, device)
    for member, metric in (
        ("MultVAE", vae_metric), ("SASRec", sas_metric), ("ensemble", ensemble_metric)
    ):
        print(
            f"{name} {member} selected_validation_HR@10={metric[0]:.6f} "
            f"NDCG@10={metric[1]:.6f}", flush=True,
        )
    with open(os.path.join(WORKING, name + "_validation.json"), "w") as handle:
        json.dump({
            "MultVAE": dict(zip(("HR@10", "NDCG@10"), vae_metric)),
            "SASRec": dict(zip(("HR@10", "NDCG@10"), sas_metric)),
            "ensemble": dict(zip(("HR@10", "NDCG@10"), ensemble_metric)),
            "evaluation_seed": 42, "n_users": sum(len(b[1]) for b in val_batches),
            "unique_negatives": 99, "fusion": "weighted_reciprocal_rank",
            "fusion_weights": {"MultVAE": 0.75, "SASRec": 0.25},
            "fusion_constant": 10.0, "fusion_rank_base": 1,
        }, handle, indent=2)
    del ensemble
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
            'trained_ensemble_module', os.path.join(_HERE, '..', 'train.py'))
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    if dataset_name not in _models:
        _models[dataset_name] = _module.load_ensemble(_HERE, dataset_name, _device)
    return _models[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    model = _load(ds)
    # Local validation and host inference share uniform-history preparation and
    # weighted reciprocal-rank fusion: MultVAE 0.75, SASRec 0.25,
    # constant 10 and one-based ranks, implemented by the shared training helper.
    return model.predict(user_id, history, candidates, dataset_name=ds)
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as handle:
        handle.write(script)


def check_history_transform():
    # Deterministic checks consume no training RNG state.
    ids, weights = transform_history([3, 1, 3])
    np.testing.assert_array_equal(ids, [1, 3])
    np.testing.assert_allclose(weights, np.full(2, 1 / math.sqrt(2)), rtol=1e-6)
    other_ids, other_weights = transform_history([1, 3])
    np.testing.assert_array_equal(ids, other_ids)
    np.testing.assert_array_equal(weights, other_weights)
    assert weights.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(weights), 1.0, rtol=1e-6)
    empty = transform_history([])
    assert empty[0].dtype == np.int64 and empty[1].dtype == np.float32
    assert empty[0].size == empty[1].size == 0
    data = Histories({0: [], 1: [4], 2: [3, 1, 3]})
    assert data.rows == [[4], [3, 1, 3]]
    packed = pack_rows([data[0], data[1], empty])
    ids, offsets, owners, input_weights = packed
    assert len(packed) == 4
    torch.testing.assert_close(ids, torch.tensor([4, 1, 3]))
    torch.testing.assert_close(offsets, torch.tensor([0, 1, 3, 3]))
    torch.testing.assert_close(owners, torch.tensor([0, 1, 1]))
    lengths = offsets[1:] - offsets[:-1]
    fallback = torch.repeat_interleave(lengths.clamp_min(1).float().rsqrt(), lengths)
    torch.testing.assert_close(input_weights, fallback)
    torch.testing.assert_close(
        prepare_sequences([[1, 2, 3, 4], [], [3]], maxlen=3),
        torch.tensor([[2, 3, 4], [0, 0, 0], [0, 0, 3]]),
    )

    class ReconstructionFixture(MultVAE):
        def __init__(self):
            nn.Module.__init__(self)
            self.decoder = nn.Identity()

        def representation(self, ids, offsets, input_weights=None):
            logits = torch.tensor([
                [0.2, 0.1, -0.2, 0.7, 0.3],
                [0.1, -0.4, 0.6, 0.2, -0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ])
            shape = (3, 2)
            return logits, torch.full(shape, 0.5), torch.full(shape, math.log(2.0))

    fixture = ReconstructionFixture()
    logits, _, _ = fixture.representation(ids, offsets)
    logprobs = F.log_softmax(logits, dim=-1)
    expected_nll = -(logprobs[0, 4] + logprobs[1, 1] + logprobs[1, 3]) / 3
    expected_kl = 1.25 - math.log(2.0)
    for anneal in (0.0, 0.07, 0.2):
        torch.testing.assert_close(
            fixture(ids, offsets, owners, anneal, input_weights),
            expected_nll + anneal * expected_kl,
        )
    a = torch.tensor([[2.0, 2.0, -1.0, 0.0]])
    b = torch.tensor([[0.0, 3.0, 3.0, -1.0]])
    torch.testing.assert_close(descending_ranks(a), torch.tensor([[0, 1, 3, 2]]))
    torch.testing.assert_close(
        rank_fusion(a, b),
        torch.tensor([[
            0.75 / 11 + 0.25 / 13,
            0.75 / 12 + 0.25 / 11,
            0.75 / 14 + 0.25 / 12,
            0.75 / 13 + 0.25 / 14,
        ]]),
    )


@torch.inference_mode()
def check_predictor_agreement(name, model, train, valid, batches, device):
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = random.Random(42).sample(users, 200)
    spec = importlib.util.spec_from_file_location(
        "ensemble_predictor_check", os.path.join(WORKING, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    # Exercise the generated loader and both saved checkpoints, without injecting
    # in-memory models. Preserve global thread settings and constructor RNG state.
    old_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            exported = predictor._load(name)
        for expected_member, actual_member in (
            (model.vae, exported.vae), (model.sasrec, exported.sasrec)
        ):
            for key, value in expected_member.state_dict().items():
                torch.testing.assert_close(
                    value.cpu(), actual_member.state_dict()[key].cpu(), rtol=0, atol=0
                )
        candidate_rows = torch.cat([batch[1] for batch in batches]).tolist()
        assert len(candidate_rows) == len(users)
        for user, candidates in zip(users, candidate_rows):
            # Matching single-user passes avoid batch-dependent rounding changes.
            packed = tuple(x.to(device) for x in pack_rows([transform_history(train[user])]))
            sequences = prepare_sequences([train[user]], model.sasrec.maxlen).to(device)
            candidates_tensor = torch.tensor([candidates], dtype=torch.long, device=device)
            expected = model.score(packed, sequences, candidates_tensor)[0].cpu().numpy()
            actual = predictor.predict(user, train[user], candidates, dataset_name=name)
            np.testing.assert_array_equal(actual, expected)
        print(f"{name} checkpoint_reload_and_predictor_agreement users={len(users)} passed", flush=True)
    finally:
        predictor._models.clear()
        torch.set_num_threads(old_threads)


def main():
    check_history_transform()
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
    # Reserve a substantial margin under the nine-hour execution ceiling.
    end = time.monotonic() + 8 * 3600
    scores = []
    for index, name in enumerate(DATASETS):
        remaining_categories = len(DATASETS) - index
        category_deadline = time.monotonic() + max(
            1.0, (end - time.monotonic()) / remaining_categories
        )
        scores.append(train_category(name, device, category_deadline))
    score = float(np.mean(scores))
    previous_scores = dict(zip(DATASETS, (0.680, 0.580, 0.440, 0.720)))
    for name, category_score in zip(DATASETS, scores):
        print(
            f"{name} selected_validation_HR@10={category_score:.6f} "
            f"previous_HR@10={previous_scores[name]:.6f} "
            f"delta={category_score - previous_scores[name]:+.6f}",
            flush=True,
        )
    # Compare measured local validation with the current reciprocal-rank result.
    # Recovery requires improvement under unchanged independent host validation.
    print(
        f"Macro validation HR@10={score:.6f} previous_HR@10=0.605000 "
        f"delta={score - 0.6050:+.6f} historical_mean_rank_HR@10=0.620000",
        flush=True,
    )
    # Report measured ensemble validation only; the host owns held-out evaluation.
    print(f'Final Validation Score: {score}')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        os.makedirs(WORKING, exist_ok=True)
        with open(os.path.join(WORKING, "training_error.txt"), "w") as handle:
            traceback.print_exc(file=handle)
        raise