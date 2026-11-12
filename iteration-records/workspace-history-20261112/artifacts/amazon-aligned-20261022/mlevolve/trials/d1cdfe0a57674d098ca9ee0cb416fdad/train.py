from __future__ import annotations

import os
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
WORK = Path(__file__).resolve().parent / "working"
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
HIDDEN = int(os.environ.get("SASREC_HIDDEN", "64"))
# Wall-clock limits are conservative relative to the nine-hour hard limit.
CATEGORY_SECONDS = float(os.environ.get("ENSEMBLE_CATEGORY_SECONDS", "1200"))
TOTAL_SECONDS = min(float(os.environ.get("ENSEMBLE_TOTAL_SECONDS", "4800")), 30000.0)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_data(name):
    train = read_split(Path(DATA_DIR) / f"{name}_train.txt")
    valid = read_split(Path(DATA_DIR) / f"{name}_valid.txt")
    stats = Path(DATA_DIR) / "dataset_stats.json"
    nitems = max(max(items, default=0) for items in train.values())
    if stats.exists():
        with open(stats) as handle:
            metadata = json.load(handle)
        nitems = int(metadata.get(name, {}).get("itemnum", nitems))
    if not train or nitems < 100:
        raise ValueError(f"Invalid training dataset: {name}")
    return train, valid, nitems


class SequentialHistories(Dataset):
    def __init__(self, train, nitems, maxlen):
        self.rows = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.seen = [set(h.tolist()) for h in self.rows]
        self.nitems = nitems
        self.maxlen = maxlen
        if not self.rows:
            raise ValueError("No eligible sequential training histories")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        history = self.rows[index]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros_like(seq)
        negative = np.zeros((self.maxlen, 64), dtype=np.int64)
        length = min(len(history) - 1, self.maxlen)
        seq[-length:] = history[-length - 1 : -1]
        pos[-length:] = history[-length:]
        if len(self.seen[index]) >= self.nitems:
            raise ValueError("No unseen negative items for a training user")
        neg = np.random.randint(1, self.nitems + 1, size=(length, 64))
        forbidden = self.seen[index]
        bad = np.fromiter(
            (int(x) in forbidden for x in neg.flat), dtype=bool, count=neg.size
        ).reshape(neg.shape)
        while bad.any():
            neg[bad] = np.random.randint(1, self.nitems + 1, size=int(bad.sum()))
            bad = np.fromiter(
                (int(x) in forbidden for x in neg.flat), dtype=bool, count=neg.size
            ).reshape(neg.shape)
        negative[-length:] = neg
        return seq, pos, negative


class Histories(Dataset):
    def __init__(self, train):
        self.rows = [
            np.unique(np.asarray(h, dtype=np.int64)) for h in train.values() if h
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def history_collate(rows):
    lengths = [len(row) for row in rows]
    items = torch.from_numpy(np.concatenate(rows))
    users = torch.repeat_interleave(torch.arange(len(rows)), torch.tensor(lengths))
    return users, items, len(rows)


def make_loader(dataset, kind):
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=True,
        collate_fn=history_collate if kind == "vae" else None,
    )


def endless(loader):
    while True:
        yield from loader


class FeedForward(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class SASRec(nn.Module):
    def __init__(self, nitems, hidden=64, blocks=2, heads=1, dropout=0.5, maxlen=128):
        super().__init__()
        self.maxlen = maxlen
        self.hidden = hidden
        self.item_emb = nn.Embedding(nitems + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.dropout = nn.Dropout(dropout)
        self.attn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks)]
        )
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                for _ in range(blocks)
            ]
        )
        self.ff_norms = nn.ModuleList(
            [nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks)]
        )
        self.ffs = nn.ModuleList([FeedForward(hidden, dropout) for _ in range(blocks)])
        self.last_norm = nn.LayerNorm(hidden, eps=1e-8)
        self.preference = nn.Linear(hidden, hidden)
        self.preference_norm = nn.LayerNorm(hidden)
        self.gate = nn.Linear(hidden * 2, hidden)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, seq):
        keep = seq.ne(0).unsqueeze(-1)
        raw = self.item_emb(seq) * keep
        length = seq.shape[1]
        positions = torch.arange(self.maxlen - length, self.maxlen, device=seq.device)
        x = self.dropout(raw * math.sqrt(self.hidden) + self.pos_emb(positions))
        x = x * keep
        causal = torch.ones(length, length, device=seq.device, dtype=torch.bool).triu(1)
        for norm, attention, fnorm, ff in zip(
            self.attn_norms, self.attentions, self.ff_norms, self.ffs
        ):
            q = norm(x)
            out, _ = attention(q, x, x, attn_mask=causal, need_weights=False)
            x = ff(fnorm(q + out)) * keep
        x = self.last_norm(x)
        prefix = raw.cumsum(1) / keep.to(raw.dtype).cumsum(1).clamp_min(1.0)
        pref = self.preference_norm(self.preference(prefix))
        gate = torch.sigmoid(self.gate(torch.cat([x, pref], dim=-1)))
        return ((1.0 - gate) * x + gate * pref) * keep

    def forward(self, seq, pos, neg):
        active = pos.ne(0)
        feat = self.log2feats(seq)[active]
        positive = (feat * self.item_emb(pos[active])).sum(-1)
        negative = torch.einsum("nd,nkd->nk", feat, self.item_emb(neg[active]))
        return positive, negative

    def predict(self, seq, candidates):
        feat = self.log2feats(seq)[:, -1]
        return (self.item_emb(candidates) * feat.unsqueeze(1)).sum(-1)


class MultVAE(nn.Module):
    def __init__(self, nitems, hidden=256, latent=64, dropout=0.5):
        super().__init__()
        self.nitems = nitems
        self.dropout = dropout
        self.encoder = nn.Linear(nitems, hidden)
        self.mu_logvar = nn.Linear(hidden, 2 * latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, nitems)
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.normal_(module.bias, std=0.001)

    def forward(self, x):
        h = F.normalize(x, p=2, dim=1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu, logvar = self.mu_logvar(torch.tanh(self.encoder(h))).chunk(2, -1)
        logvar = logvar.clamp(-12, 12)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        logits = self.decoder(z)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return logits, kl


def pack_history(history, maxlen, device):
    seq = torch.zeros((1, maxlen), dtype=torch.long, device=device)
    hist = history[-maxlen:]
    if hist:
        seq[0, -len(hist) :] = torch.tensor(hist, dtype=torch.long, device=device)
    return seq


def rank_scores(scores, candidates):
    # Item-ID tie breaking does not depend on where the target was placed.
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.asarray(candidates), -scores))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


@torch.inference_mode()
def score_members(models, history, candidates, nitems, maxlen, device):
    scores = {}
    cand = torch.tensor([candidates], dtype=torch.long, device=device)
    if "sas" in models:
        seq = pack_history(history, maxlen, device)
        scores["sas"] = models["sas"].predict(seq, cand)[0].float().cpu().numpy()
    if "vae" in models:
        x = torch.zeros((1, nitems), device=device)
        if history:
            ids = torch.tensor(sorted(set(history)), dtype=torch.long, device=device)
            x[0, ids - 1] = 1.0
        logits, _ = models["vae"](x)
        scores["vae"] = logits[0, cand[0] - 1].float().cpu().numpy()
    return scores


def fused_rank_scores(ranks, sas_weight=0.5):
    # Keep single-member checkpoint evaluation independent of fusion weights.
    if len(ranks) == 1:
        return 1.0 / (60.0 + next(iter(ranks.values())))
    return (
        sas_weight / (60.0 + ranks["sas"])
        + (1.0 - sas_weight) / (60.0 + ranks["vae"])
    )


def fused_scores(member_scores, candidates, sas_weight=0.5):
    ranks = {
        kind: rank_scores(member_scores[kind], candidates)
        for kind in ("sas", "vae") if kind in member_scores
    }
    return fused_rank_scores(ranks, sas_weight)


def select_sas_weight(hits):
    # Integer hits decide quality; ties prefer equal fusion, then lower weight.
    return min(hits, key=lambda w: (-hits[w], abs(w - 0.5), w))


@torch.inference_mode()
def select_fusion(models, cases, nitems, maxlen, device):
    for model in models.values():
        model.eval()
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    hits = {w: 0 for w in weights}
    ndcg = {w: 0.0 for w in weights}
    member_hits = {kind: 0 for kind in ("sas", "vae")}
    member_ndcg = {kind: 0.0 for kind in ("sas", "vae")}
    # Both checkpoints are already restored. Cache ranks with one scoring pass
    # per case, then search only the five specified weights at constant 60.
    cached = []
    for history, candidates in cases:
        outputs = score_members(models, history, candidates, nitems, maxlen, device)
        ranks = {
            kind: rank_scores(outputs[kind], candidates)
            for kind in ("sas", "vae")
        }
        cached.append((candidates, ranks))
        for kind in ("sas", "vae"):
            rank = int(ranks[kind][0])
            if rank <= 10:
                member_hits[kind] += 1
                member_ndcg[kind] += 1.0 / math.log2(rank + 1)
    for candidates, ranks in cached:
        for weight in weights:
            scores = fused_rank_scores(ranks, weight)
            rank = int(rank_scores(scores, candidates)[0])
            if rank <= 10:
                hits[weight] += 1
                ndcg[weight] += 1.0 / math.log2(rank + 1)
    weight = select_sas_weight(hits)
    assert hits[weight] >= hits[0.5]
    metrics = {
        kind: (member_hits[kind] / len(cases), member_ndcg[kind] / len(cases))
        for kind in ("sas", "vae")
    }
    metrics["ensemble"] = (hits[weight] / len(cases), ndcg[weight] / len(cases))
    diagnostics = {
        str(w): {"hits": hits[w], "HR@10": hits[w] / len(cases)}
        for w in weights
    }
    return weight, metrics, diagnostics


def verify_negative_supervision():
    # Check 64 uniform negatives and the existing 65-way loss on a tiny CPU
    # fixture. Restore RNG states so these checks do not change training.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        np.random.seed(0)
        torch.random.default_generator.manual_seed(42)
        dataset = SequentialHistories(
            {1: [1, 2, 3, 4, 5, 1], 2: [2, 3]}, nitems=8, maxlen=3
        )
        samples = [dataset[index] for index in range(len(dataset))]
        for index, (seq, pos, negative) in enumerate(samples):
            assert seq.shape == pos.shape == (3,)
            assert negative.shape == (3, 64)
            active = pos != 0
            assert np.all(seq[~active] == 0)
            assert np.all(negative[~active] == 0)
            assert np.all((negative[active] >= 1) & (negative[active] <= 8))
            assert not set(negative[active].ravel()).intersection(dataset.seen[index])
        # Items 1 and 2 fall outside the first retained input window but must
        # still be excluded. Only training history enters that exclusion set.
        np.testing.assert_array_equal(samples[0][0], [3, 4, 5])
        np.testing.assert_array_equal(samples[0][1], [4, 5, 1])
        np.testing.assert_array_equal(samples[1][0], [0, 0, 2])
        np.testing.assert_array_equal(samples[1][1], [0, 0, 3])
        assert np.any(samples[0][2] == 8)
        assert np.unique(samples[0][2][-1]).size < 64

        seq, pos, negative = [
            torch.from_numpy(np.stack(values)) for values in zip(*samples)
        ]
        model = SASRec(
            nitems=8, hidden=8, blocks=1, heads=1, dropout=0.0, maxlen=3
        )
        positive, negative = model(seq, pos, negative)
        count = int(pos.ne(0).sum())
        assert positive.shape == (count,)
        assert negative.shape == (count, 64)
        logits = torch.cat([positive[:, None], negative], dim=1).float()
        assert logits.shape == (count, 65)
        loss = F.cross_entropy(logits, torch.zeros(count, dtype=torch.long))
        reference = (torch.logsumexp(logits, dim=1) - logits[:, 0]).mean()
        torch.testing.assert_close(loss, reference)
        assert torch.isfinite(loss).item()
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all().item()
        assert model.item_emb.weight.grad.abs().sum().item() > 0
        assert model.attentions[0].in_proj_weight.grad.abs().sum().item() > 0
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def verify_fusion():
    candidates = np.array([9, 2, 7, 4])
    members = {
        "vae": np.array([0.0, 1.0, 1.0, 0.0]),
        "sas": np.array([2.0, 2.0, 0.0, 1.0]),
    }
    sas_ranks = rank_scores(members["sas"], candidates)
    vae_ranks = rank_scores(members["vae"], candidates)
    np.testing.assert_array_equal(sas_ranks, [2, 1, 4, 3])
    np.testing.assert_array_equal(vae_ranks, [4, 1, 2, 3])
    np.testing.assert_allclose(
        fused_scores(members, candidates, 1.0), 1.0 / (60.0 + sas_ranks)
    )
    np.testing.assert_allclose(
        fused_scores(members, candidates, 0.0), 1.0 / (60.0 + vae_ranks)
    )
    expected = (1.0 / (60.0 + sas_ranks) + 1.0 / (60.0 + vae_ranks)) / 2
    np.testing.assert_array_equal(fused_scores(members, candidates), expected)
    for kind in ("sas", "vae"):
        np.testing.assert_array_equal(
            fused_scores({kind: members[kind]}, candidates, 0.0),
            1.0 / (60.0 + rank_scores(members[kind], candidates)),
        )
    permutation = np.array([2, 0, 3, 1])
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        original = fused_scores(members, candidates, weight)
        permuted = fused_scores(
            {kind: scores[permutation] for kind, scores in members.items()},
            candidates[permutation], weight,
        )
        np.testing.assert_array_equal(permuted, original[permutation])
    assert select_sas_weight({w: 3 for w in (0.0, 0.25, 0.5, 0.75, 1.0)}) == 0.5
    assert select_sas_weight({0.0: 2, 0.25: 3, 0.5: 2, 0.75: 3, 1.0: 2}) == 0.25


def validation_cases(train, valid, nitems):
    rng = random.Random(42)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for u in users:
        blocked = set(train[u]) | set(valid[u])
        if nitems - len(blocked) < 99:
            raise ValueError("Cannot generate 99 unique unseen candidates")
        negative = set()
        while len(negative) < 99:
            item = rng.randint(1, nitems)
            if item not in blocked:
                negative.add(item)
        cases.append((train[u], [valid[u][0]] + list(negative)))
    if not cases:
        raise ValueError("Empty validation split")
    return cases


@torch.inference_mode()
def evaluate(models, cases, nitems, maxlen, device, sas_weight=0.5):
    for model in models.values():
        model.eval()
    totals = {key: [0.0, 0.0] for key in list(models) + ["ensemble"]}
    for history, candidates in cases:
        outputs = score_members(models, history, candidates, nitems, maxlen, device)
        outputs["ensemble"] = fused_scores(outputs, candidates, sas_weight)
        for key, scores in outputs.items():
            rank = int(rank_scores(scores, candidates)[0])
            if rank <= 10:
                totals[key][0] += 1.0
                totals[key][1] += 1.0 / math.log2(rank + 1)
    return {
        key: (hits / len(cases), ndcg / len(cases))
        for key, (hits, ndcg) in totals.items()
    }


def fit_member(name, kind, train, cases, config, device, deadline):
    model = (SASRec(**config) if kind == "sas" else MultVAE(**config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    dataset = (
        SequentialHistories(train, config["nitems"], config["maxlen"])
        if kind == "sas"
        else Histories(train)
    )
    loader = make_loader(dataset, kind)
    batches = endless(loader)
    checkpoint = WORK / f"{name}_{kind}.pt"
    best = (-1.0, -1.0)
    best_epoch = 0
    updates = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        completed = 0
        for _ in range(STEPS):
            if completed and time.monotonic() >= deadline:
                break
            batch = next(batches)
            if kind == "sas":
                seq, pos, negative = [x.to(device, non_blocking=True) for x in batch]
                # Remove only columns that are padding for every row; absolute
                # positional indices remain identical to deployed inference.
                length = int(pos.ne(0).sum(1).max().item())
                seq, pos, negative = (
                    seq[:, -length:],
                    pos[:, -length:],
                    negative[:, -length:],
                )
                positive, negative = model(seq, pos, negative)
                logits = torch.cat([positive[:, None], negative], dim=1)
                loss = F.cross_entropy(
                    logits.float(),
                    torch.zeros(len(logits), device=device, dtype=torch.long),
                )
            else:
                rows, items, count = batch
                x = torch.zeros((count, config["nitems"]), device=device)
                x[rows.to(device), items.to(device) - 1] = 1.0
                logits, kl = model(x)
                reconstruction = -(F.log_softmax(logits, dim=1) * x).sum(1).mean()
                anneal = min(0.2, updates / 10000.0)
                loss = reconstruction + anneal * kl
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {name}/{kind}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
            completed += 1
            total_loss += float(loss.detach())
        expired = time.monotonic() >= deadline
        checked = epoch == 1 or epoch % 5 == 0 or expired or epoch == EPOCHS
        suffix = ""
        if checked:
            metric = evaluate({kind: model}, cases, config["nitems"], MAXLEN, device)[
                kind
            ]
            suffix = f" HR@10={metric[0]:.4f} NDCG@10={metric[1]:.4f}"
            if metric > best:
                best, best_epoch = metric, epoch
                torch.save(model.state_dict(), checkpoint)
        print(
            f"{name}/{kind} epoch={epoch} loss={total_loss / max(completed, 1):.5f}"
            + suffix,
            flush=True,
        )
        if expired or epoch - best_epoch >= 25:
            break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    del batches, loader, optimizer
    return model


def load_ensemble(name, directory, device):
    directory = Path(directory)
    with open(directory / f"{name}_meta.json") as handle:
        meta = json.load(handle)
    models = {}
    for kind in ["sas", "vae"]:
        config = meta[kind]
        model = (SASRec(**config) if kind == "sas" else MultVAE(**config)).to(device)
        state = torch.load(
            directory / f"{name}_{kind}.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        models[kind] = model
    return models, meta


PREDICTOR = r"""from pathlib import Path
import importlib.util
import torch

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('rec_ensemble_training', _HERE.parent / 'train.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_loaded_name = None
_models = None
_meta = None

def predict(user_id, history, candidates, *, dataset_name=None):
    global _loaded_name, _models, _meta
    name = dataset_name if dataset_name is not None else _module.DATASETS[0]
    if name not in _module.DATASETS:
        raise ValueError(f'Unknown dataset: {name}')
    if _loaded_name != name:
        _models = None
        if _device.type == 'cuda':
            torch.cuda.empty_cache()
        _models, _meta = _module.load_ensemble(name, _HERE, _device)
        _loaded_name = name
    with torch.inference_mode():
        members = _module.score_members(_models, list(history), list(candidates),
                                        _meta['sas']['nitems'],
                                        _meta['sas']['maxlen'], _device)
        return _module.fused_scores(
            members, candidates, sas_weight=_meta.get('sas_weight', 0.5)
        ).tolist()
"""


def main():
    seed_all(42)
    verify_negative_supervision()
    verify_fusion()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "predict.py").write_text(PREDICTOR)
    overall_deadline = time.monotonic() + TOTAL_SECONDS
    scores = []
    print(
        "Validation: supplied validation split; up to 200 users/category; "
        "seed=42; one held-out next item against 99 unique unseen negatives; "
        "HR@10 is mean(target rank <= 10); final score is category macro mean.",
        flush=True,
    )
    for index, name in enumerate(DATASETS):
        train, valid, nitems = load_data(name)
        cases = validation_cases(train, valid, nitems)
        remaining_categories = len(DATASETS) - index
        allowance = min(
            CATEGORY_SECONDS,
            max(30.0, (overall_deadline - time.monotonic()) / remaining_categories),
        )
        start = time.monotonic()
        sas_config = dict(
            nitems=nitems,
            hidden=HIDDEN,
            blocks=int(os.environ.get("SASREC_BLOCKS", "2")),
            heads=int(os.environ.get("SASREC_HEADS", "1")),
            dropout=float(os.environ.get("SASREC_DROPOUT", "0.5")),
            maxlen=MAXLEN,
        )
        vae_config = dict(nitems=nitems, hidden=256, latent=64, dropout=0.5)
        metadata = {"sas": sas_config, "vae": vae_config}
        with open(WORK / f"{name}_meta.json", "w") as handle:
            json.dump(metadata, handle)
        sas = fit_member(
            name, "sas", train, cases, sas_config, device, start + allowance * 0.55
        )
        # Keep the selected SASRec on CPU while training the independent VAE.
        sas.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        vae = fit_member(
            name, "vae", train, cases, vae_config, device, start + allowance * 0.95
        )
        models = {"sas": sas.to(device), "vae": vae}
        sas_weight, metrics, diagnostics = select_fusion(
            models, cases, nitems, MAXLEN, device
        )
        metadata["sas_weight"] = sas_weight
        metadata["fusion_validation"] = diagnostics
        with open(WORK / f"{name}_meta.json", "w") as handle:
            json.dump(metadata, handle)
        scores.append(metrics["ensemble"][0])
        print(
            f'{name}: SASRec={metrics["sas"][0]:.4f} '
            f'MultVAE={metrics["vae"][0]:.4f} '
            f'sas_weight={sas_weight:g} '
            f'selected local fusion HR@10={metrics["ensemble"][0]:.4f}; '
            f'all five weights={json.dumps(diagnostics, sort_keys=True)}',
            flush=True,
        )
        # Verify checkpoint reconstruction with exactly the same deployed input.
        history, candidates = cases[0]
        reference = fused_scores(
            score_members(models, history, candidates, nitems, MAXLEN, device),
            candidates,
            sas_weight=sas_weight,
        )
        del models, sas, vae
        if device.type == "cuda":
            torch.cuda.empty_cache()
        reloaded, meta = load_ensemble(name, WORK, device)
        assert meta["sas_weight"] == sas_weight
        restored = fused_scores(
            score_members(reloaded, history, candidates, nitems, MAXLEN, device),
            candidates,
            sas_weight=meta["sas_weight"],
        )
        np.testing.assert_allclose(reference, restored, rtol=0, atol=1e-10)
        del reloaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Execute the actual exported predictor and exercise its checkpoint
        # loading, dataset routing, saved weight, and candidate ordering.
        predictor_path = WORK / "predict.py"
        predictor = {"__file__": str(predictor_path)}
        exec(compile(predictor_path.read_text(), str(predictor_path), "exec"), predictor)
        deployed = predictor["predict"](0, history, candidates, dataset_name=name)
        np.testing.assert_allclose(reference, deployed, rtol=0, atol=1e-10)
        permutation = np.arange(len(candidates))[::-1]
        permuted_candidates = [candidates[i] for i in permutation]
        deployed_permuted = predictor["predict"](
            0, history, permuted_candidates, dataset_name=name
        )
        np.testing.assert_allclose(
            reference[permutation], deployed_permuted, rtol=0, atol=1e-10
        )
        predictor.clear()
        del predictor, train, valid
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Held-out files are deliberately left to the independent host evaluator.
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")
    print(
        "This is the locally selected four-category macro validation HR@10. "
        "The prior independently measured host macro HR@10 is 0.5625 "
        "(Movies_and_TV=0.595, Industrial_and_Scientific=0.615, "
        "Electronics=0.370, CDs_and_Vinyl=0.670); "
        "the original Electronics baseline is 0.420. "
        "New host category scores and improvement remain unmeasured until "
        "independent host validation; held-out evaluation is host-only.",
        flush=True,
    )


if __name__ == "__main__":
    main()
