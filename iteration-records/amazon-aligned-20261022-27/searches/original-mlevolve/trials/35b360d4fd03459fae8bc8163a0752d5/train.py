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
        # MultiDAE: deterministic latent projection, without a residual bypass.
        self.latent = nn.Linear(hidden, latent)
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
        z = self.latent(torch.tanh(self.encoder(h)))
        logits = self.decoder(z)
        return logits, logits.new_zeros(())


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


def fuse_ranks(ranks, sas_weight=0.5):
    if len(ranks) == 1:
        return 1.0 / (60.0 + next(iter(ranks.values())))
    return (
        sas_weight / (60.0 + ranks["sas"])
        + (1.0 - sas_weight) / (60.0 + ranks["vae"])
    )


def fused_scores(member_scores, candidates, sas_weight=0.5):
    ranks = {
        kind: rank_scores(scores, candidates)
        for kind, scores in member_scores.items()
    }
    return fuse_ranks(ranks, sas_weight)


FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def select_fusion_result(results):
    return min(
        results,
        key=lambda result: (
            -result["hits"],
            abs(result["sas_weight"] - 0.5),
            result["sas_weight"],
        ),
    )


@torch.inference_mode()
def cache_validation_ranks(models, cases, nitems, maxlen, device):
    for model in models.values():
        model.eval()
    cached = []
    for history, candidates in cases:
        outputs = score_members(models, history, candidates, nitems, maxlen, device)
        cached.append((candidates, {
            kind: rank_scores(scores, candidates)
            for kind, scores in outputs.items()
        }))
    return cached


def metrics_from_ranks(ranks):
    hits = sum(int(rank <= 10) for rank in ranks)
    ndcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    return {
        "hits": hits,
        "n_users": len(ranks),
        "HR@10": hits / len(ranks),
        "NDCG@10": ndcg / len(ranks),
    }


def select_fusion(cached):
    results = []
    for sas_weight in FUSION_WEIGHTS:
        ranks = [
            int(rank_scores(fuse_ranks(member_ranks, sas_weight), candidates)[0])
            for candidates, member_ranks in cached
        ]
        results.append({
            "sas_weight": sas_weight,
            "vae_weight": 1.0 - sas_weight,
            **metrics_from_ranks(ranks),
        })
    return select_fusion_result(results), results


def check_fusion():
    # Fixed arrays and permutations never advance training RNG streams.
    candidates = np.asarray([17, 3, 29, 11, 5])
    outputs = {
        "sas": np.asarray([2.0, 2.0, -1.0, 4.0, 0.0]),
        "vae": np.asarray([0.0, 3.0, 3.0, -1.0, 2.0]),
    }
    ranks = {kind: rank_scores(s, candidates) for kind, s in outputs.items()}
    np.testing.assert_array_equal(ranks["sas"], [3, 2, 5, 1, 4])
    np.testing.assert_array_equal(ranks["vae"], [4, 1, 2, 5, 3])
    legacy = sum(1.0 / (60.0 + r) for r in ranks.values()) / len(ranks)
    np.testing.assert_array_equal(fused_scores(outputs, candidates), legacy)
    for sas_weight in FUSION_WEIGHTS:
        expected = (
            sas_weight / (60.0 + ranks["sas"])
            + (1.0 - sas_weight) / (60.0 + ranks["vae"])
        )
        actual = fused_scores(outputs, candidates, sas_weight)
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(fuse_ranks(ranks, sas_weight), actual)
        np.testing.assert_array_equal(
            fused_scores(dict(reversed(list(outputs.items()))), candidates, sas_weight),
            actual,
        )
        for permutation in (np.arange(len(candidates))[::-1], np.asarray([2, 0, 4, 1, 3])):
            permuted = fused_scores(
                {kind: s[permutation] for kind, s in outputs.items()},
                candidates[permutation], sas_weight,
            )
            np.testing.assert_array_equal(permuted, actual[permutation])
            np.testing.assert_array_equal(
                rank_scores(permuted, candidates[permutation]),
                rank_scores(actual, candidates)[permutation],
            )
        for kind in outputs:
            np.testing.assert_array_equal(
                fused_scores({kind: outputs[kind]}, candidates, sas_weight),
                1.0 / (60.0 + ranks[kind]),
            )
    np.testing.assert_array_equal(
        fused_scores(outputs, candidates, 0.0), 1.0 / (60.0 + ranks["vae"])
    )
    np.testing.assert_array_equal(
        fused_scores(outputs, candidates, 1.0), 1.0 / (60.0 + ranks["sas"])
    )
    for hits, expected in (
        ([1, 1, 1, 1, 1], 0.5),
        ([0, 2, 1, 2, 0], 0.25),
        ([2, 0, 0, 0, 2], 0.0),
        ([0, 0, 1, 0, 2], 1.0),
    ):
        results = [
            {"sas_weight": weight, "hits": count}
            for weight, count in zip(FUSION_WEIGHTS, hits)
        ]
        assert select_fusion_result(results)["sas_weight"] == expected
        assert select_fusion_result(list(reversed(results)))["sas_weight"] == expected
    selected, results = select_fusion([(candidates, ranks)])
    assert len(results) == 5
    assert selected["sas_weight"] == 0.5


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


def check_sas_negatives():
    # Exercise padding and truncated histories without advancing training RNGs.
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            np.random.seed(0)
            torch.manual_seed(42)
            dataset = SequentialHistories(
                {1: [1, 2, 3], 2: [4, 5, 6, 7, 8, 9, 10]},
                nitems=128,
                maxlen=4,
            )
            rows = [dataset[index] for index in range(len(dataset))]
            for index, (seq, pos, negative) in enumerate(rows):
                history = dataset.rows[index]
                length = min(len(history) - 1, dataset.maxlen)
                assert negative.shape == (dataset.maxlen, 64)
                assert np.all(seq[:-length] == 0)
                assert np.all(pos[:-length] == 0)
                assert np.all(negative[:-length] == 0)
                np.testing.assert_array_equal(seq[-length:], history[-length - 1 : -1])
                np.testing.assert_array_equal(pos[-length:], history[-length:])
                assert np.all(negative[-length:] >= 1)
                assert np.all(negative[-length:] <= dataset.nitems)
                # Exclude the full history, including items outside the window.
                assert not np.isin(negative[-length:], history).any()
            seq, pos, negative = [
                torch.from_numpy(np.stack(arrays)) for arrays in zip(*rows)
            ]
            model = SASRec(
                nitems=128, hidden=8, blocks=1, heads=1, dropout=0.0, maxlen=4
            )
            positive, negative = model(seq, pos, negative)
            logits = torch.cat([positive[:, None], negative], dim=1)
            assert logits.shape == (int(pos.ne(0).sum()), 65)
            loss = F.cross_entropy(
                logits.float(), torch.zeros(len(logits), dtype=torch.long)
            )
            assert torch.isfinite(loss).item()
            loss.backward()
            for parameter in model.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all().item()
            assert model.item_emb.weight.grad.abs().sum().item() > 0
            assert model.attentions[0].in_proj_weight.grad.abs().sum().item() > 0
    finally:
        np.random.set_state(numpy_state)


def fit_member(name, kind, train, cases, config, device, deadline):
    fit_start = time.monotonic()
    # Each expert starts its own seed-42 stream; other experts cannot advance it.
    # DataLoader shuffling and worker sampling already use independent generators.
    seed_all(42)
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
                logits, _ = model(x)
                reconstruction = -(F.log_softmax(logits, dim=1) * x).sum(1).mean()
                loss = reconstruction
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
            f" completed_updates={completed} cumulative_updates={updates}"
            f" elapsed_seconds={time.monotonic() - fit_start:.3f}"
            + suffix,
            flush=True,
        )
        if expired or epoch - best_epoch >= 25:
            break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval()
    del batches, loader, optimizer
    return model


def load_ensemble(name, directory, device):
    directory = Path(directory)
    with open(directory / f"{name}_meta.json") as handle:
        meta = json.load(handle)
    if meta.get("architectures", {}).get("vae") != "MultiDAE":
        raise ValueError(f"Expected a MultiDAE checkpoint configuration: {name}")
    models = {}
    for kind in ["sas", "vae"]:
        config = meta[kind]
        # Checkpoint reconstruction must not consume the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            model = (SASRec(**config) if kind == "sas" else MultVAE(**config)).to(device)
        state = torch.load(
            directory / f"{name}_{kind}.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)
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
        if _meta.get('architectures', {}).get('vae') != 'MultiDAE':
            raise ValueError(f'Expected MultiDAE architecture: {name}')
        _loaded_name = name
    with torch.inference_mode():
        members = _module.score_members(_models, list(history), list(candidates),
                                        _meta['sas']['nitems'],
                                        _meta['sas']['maxlen'], _device)
        return _module.fused_scores(members, candidates,
                                    _meta.get('sas_weight', 0.5)).tolist()
"""


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "predict.py").write_text(PREDICTOR)
    overall_deadline = time.monotonic() + TOTAL_SECONDS
    check_sas_negatives()
    check_fusion()
    scores = []
    equal_scores = []
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
        metadata = {
            "sas": sas_config,
            "vae": vae_config,
            "architectures": {"sas": "SASRec", "vae": "MultiDAE"},
        }
        with open(WORK / f"{name}_meta.json", "w") as handle:
            json.dump(metadata, handle)
        sas = fit_member(
            name, "sas", train, cases, sas_config, device, start + allowance * 0.55
        )
        # Keep the selected SASRec on CPU while training the independent MultiDAE.
        sas.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        vae = fit_member(
            name, "vae", train, cases, vae_config, device, start + allowance * 0.95
        )
        # Select only after strictly reloading both independently chosen checkpoints.
        del sas, vae
        if device.type == "cuda":
            torch.cuda.empty_cache()
        models, meta = load_ensemble(name, WORK, device)
        assert all(not model.training for model in models.values())
        assert all(not p.requires_grad for model in models.values() for p in model.parameters())
        cached = cache_validation_ranks(models, cases, nitems, MAXLEN, device)
        selected, grid_results = select_fusion(cached)
        sas_weight = selected["sas_weight"]
        equal_result = next(r for r in grid_results if r["sas_weight"] == 0.5)
        assert selected["hits"] >= equal_result["hits"]
        member_metrics = {
            kind: metrics_from_ranks([int(ranks[kind][0]) for _, ranks in cached])
            for kind in ("sas", "vae")
        }
        metadata.update({
            "sas_weight": sas_weight,
            "vae_weight": 1.0 - sas_weight,
            "rrf_constant": 60.0,
            "fusion_grid_results": grid_results,
            "selected_metrics": dict(selected),
            "member_metrics": member_metrics,
            "equal_weight_metrics": dict(equal_result),
        })
        with open(WORK / f"{name}_meta.json", "w") as handle:
            json.dump(metadata, handle, indent=2)
        metrics = {
            kind: (result["HR@10"], result["NDCG@10"])
            for kind, result in member_metrics.items()
        }
        metrics["ensemble"] = (selected["HR@10"], selected["NDCG@10"])
        scores.append(metrics["ensemble"][0])
        equal_scores.append(equal_result["HR@10"])
        print(
            f'{name}: SASRec={metrics["sas"][0]:.4f} '
            f'MultiDAE={metrics["vae"][0]:.4f} '
            f'sas_weight={sas_weight:.2f} vae_weight={1.0 - sas_weight:.2f} '
            f'local equal HR@10={equal_result["HR@10"]:.4f} '
            f'local selected HR@10={metrics["ensemble"][0]:.4f} '
            f'hits={selected["hits"]}/{selected["n_users"]}',
            flush=True,
        )
        print(f'{name}: expert metrics={json.dumps(member_metrics)}', flush=True)
        print(f'{name}: selected ensemble metrics={json.dumps(selected)}', flush=True)
        print(f'{name}: fusion grid={json.dumps(grid_results)}', flush=True)
        # Reuse the first cached case for all deterministic deployment checks.
        history, candidates = cases[0]
        reference_members = score_members(
            models, history, candidates, nitems, MAXLEN, device
        )
        repeated_members = score_members(
            models, history, candidates, nitems, MAXLEN, device
        )
        for kind in reference_members:
            np.testing.assert_array_equal(reference_members[kind], repeated_members[kind])
            np.testing.assert_array_equal(
                rank_scores(reference_members[kind], candidates), cached[0][1][kind]
            )
        reference = fused_scores(reference_members, candidates, sas_weight)
        references = {
            weight: fused_scores(reference_members, candidates, weight)
            for weight in FUSION_WEIGHTS
        }
        del models, cached
        if device.type == "cuda":
            torch.cuda.empty_cache()
        reloaded, meta = load_ensemble(name, WORK, device)
        assert meta["sas_weight"] == sas_weight
        assert meta["vae_weight"] == 1.0 - sas_weight
        assert meta["rrf_constant"] == 60.0
        assert meta["fusion_grid_results"] == grid_results
        restored_members = score_members(
            reloaded, history, candidates, nitems, MAXLEN, device
        )
        assert meta["architectures"]["vae"] == "MultiDAE"
        for kind in reference_members:
            np.testing.assert_array_equal(reference_members[kind], restored_members[kind])
        restored = fused_scores(restored_members, candidates, meta["sas_weight"])
        np.testing.assert_allclose(reference, restored, rtol=0, atol=1e-10)
        for weight in FUSION_WEIGHTS:
            np.testing.assert_allclose(
                references[weight], fused_scores(restored_members, candidates, weight),
                rtol=0, atol=1e-10,
            )
        del reloaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Load the written predictor and verify the deployed dataset route.
        import importlib.util

        predictor_spec = importlib.util.spec_from_file_location(
            "generated_predictor_check", WORK / "predict.py"
        )
        predictor = importlib.util.module_from_spec(predictor_spec)
        predictor_spec.loader.exec_module(predictor)
        deployed = predictor.predict(
            0, history, candidates, dataset_name=name
        )
        np.testing.assert_allclose(reference, deployed, rtol=0, atol=1e-10)
        assert predictor._loaded_name == name
        assert predictor._meta["architectures"]["vae"] == "MultiDAE"
        exported_members = predictor._module.score_members(
            predictor._models, history, candidates, nitems, MAXLEN, predictor._device
        )
        for kind in reference_members:
            np.testing.assert_array_equal(reference_members[kind], exported_members[kind])
        # The category loop exercises every saved dataset route and all weights.
        permutation = np.arange(len(candidates))[::-1]
        permuted_candidates = np.asarray(candidates)[permutation].tolist()
        for weight in FUSION_WEIGHTS:
            predictor._meta["sas_weight"] = weight
            deployed = predictor.predict(0, history, candidates, dataset_name=name)
            np.testing.assert_allclose(references[weight], deployed, rtol=0, atol=1e-10)
            permuted = predictor.predict(
                0, history, permuted_candidates, dataset_name=name
            )
            np.testing.assert_allclose(
                references[weight][permutation], permuted, rtol=0, atol=1e-10
            )
            np.testing.assert_array_equal(
                rank_scores(permuted, permuted_candidates),
                rank_scores(references[weight], candidates)[permutation],
            )
        # Missing weights in legacy metadata retain equal-weight behavior.
        predictor._meta.pop("sas_weight")
        deployed = predictor.predict(0, history, candidates, dataset_name=name)
        np.testing.assert_allclose(references[0.5], deployed, rtol=0, atol=1e-10)
        predictor._meta["sas_weight"] = sas_weight
        if name == DATASETS[0]:
            np.testing.assert_allclose(
                reference, predictor.predict(0, history, candidates),
                rtol=0, atol=1e-10,
            )
        predictor._models = None
        del predictor, predictor_spec, train, valid
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Held-out files are deliberately left to the independent host evaluator.
    score = float(np.mean(scores))
    equal_score = float(np.mean(equal_scores))
    print(
        f"Local macro HR@10: selected={score:.6f}, equal={equal_score:.6f}; "
        "prior independent host macro HR@10: latest=0.5637, best=0.5737. "
        "This run's independent host validation is pending; compare its reported "
        "macro HR@10 with both references before concluding improvement.",
        flush=True,
    )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
