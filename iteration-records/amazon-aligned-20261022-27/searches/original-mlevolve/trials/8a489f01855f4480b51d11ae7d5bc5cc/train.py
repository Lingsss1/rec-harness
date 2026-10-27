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
        return (x * self.item_emb(pos)).sum(-1), (x * self.item_emb(neg)).sum(-1)

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
        self.enc_bias = nn.Parameter(torch.zeros(hidden))
        self.posterior = nn.Linear(hidden, 2 * latent)
        self.decoder_hidden = nn.Linear(latent, hidden)
        self.output = nn.Linear(hidden, itemnum + 1)
        nn.init.normal_(self.encoder.weight, std=0.02)
        nn.init.xavier_normal_(self.output.weight)
        with torch.no_grad():
            self.encoder.weight[0].zero_()

    def representation(self, ids, offsets, counts):
        # Unique-item bags have L2 norm sqrt(number of observed items).
        if self.training:
            ids = ids * (torch.rand(ids.shape, device=ids.device) >= self.dropout)
        encoded = self.encoder(ids, offsets) / counts.clamp_min(1).sqrt().unsqueeze(-1)
        if self.training:
            encoded = encoded / (1.0 - self.dropout)
        encoded = torch.tanh(encoded + self.enc_bias)
        mu, logvar = self.posterior(encoded).chunk(2, -1)
        logvar = logvar.clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        h = torch.tanh(self.decoder_hidden(z)) + encoded
        return h, mu, logvar

    def forward(self, ids, offsets, counts):
        h, mu, logvar = self.representation(ids, offsets, counts)
        logits = self.output(h)
        logits = logits.masked_fill(
            torch.arange(self.itemnum + 1, device=ids.device)[None] == 0, -1e9
        )
        return logits, mu, logvar

    def predict(self, ids, offsets, counts, candidates):
        h, _, _ = self.representation(ids, offsets, counts)
        return (self.output.weight[candidates] * h.unsqueeze(1)).sum(
            -1
        ) + self.output.bias[candidates]


def bag_tensors(histories, device):
    bags = [sorted(set(h)) or [0] for h in histories]
    lengths = [len(h) for h in bags]
    ids = torch.tensor([i for h in bags for i in h], dtype=torch.long, device=device)
    offsets = torch.tensor(
        np.cumsum([0] + lengths[:-1]), dtype=torch.long, device=device
    )
    counts = torch.tensor(
        [len(set(h) - {0}) for h in bags], dtype=torch.float32, device=device
    )
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
        out = model.predict(*bag_tensors(histories, device), c)
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
            logits, mu, logvar = model(*bag_tensors(inputs, device))
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
            torch.save({"config": config, "state": model.state_dict()}, path)
        else:
            stale += 1
        if stale >= 20 or time.monotonic() >= deadline:
            break
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)["state"]
    )
    return model.eval()


def train_sas(ds, train, n, cases, device, deadline):
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
            neg = rng.randint(1, n + 1, size=pos.shape)
            for row, idx in enumerate(indices):
                for col in np.flatnonzero(pos[row]):
                    while int(neg[row, col]) in seen[idx]:
                        neg[row, col] = rng.randint(1, n + 1)
            seq_t = torch.tensor(seq, device=device)
            pos_t = torch.tensor(pos, device=device)
            neg_t = torch.tensor(neg, device=device)
            a, b = model(seq_t, pos_t, neg_t)
            keep = pos_t.ne(0)
            loss = (
                nn.functional.softplus(-a[keep]).mean()
                + nn.functional.softplus(b[keep]).mean()
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Wall-clock cap leaves substantial room below the nine-hour hard limit.
    total_seconds = min(float(os.environ.get("TRAIN_TIME_LIMIT", "24000")), 28000.0)
    end = time.monotonic() + total_seconds
    scores = []
    write_predict_script()
    print(
        "Validation: supplied train/valid splits; seed=42; up to 200 users/category; "
        "99 unique unseen negatives; HR@10=mean(target rank<=10); macro-average over four categories.",
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
        del mod, predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
