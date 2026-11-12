from __future__ import annotations

import os
import json
import math
import random
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "100"))
DIM = int(os.environ.get("SASREC_HIDDEN", "96"))
LR = float(os.environ.get("SASREC_LR", "0.001"))
DROPOUT = float(os.environ.get("SASREC_DROPOUT", "0.2"))
NEGATIVES = 32
# Reserve time for saving and predictor creation within the nine-hour limit.
RUN_SECONDS = float(os.environ.get("CNN_RUN_SECONDS", "28800"))


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
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    itemnum = max(max(v) for v in train.values() if v)
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if name in stats:
            itemnum = int(stats[name]["itemnum"])
    return train, valid, itemnum


def pack_history(history, maxlen):
    out = np.zeros(maxlen, dtype=np.int64)
    history = history[-maxlen:]
    if history:
        out[-len(history) :] = history
    return out


class Histories(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [
            np.asarray(v, dtype=np.int64) for v in train.values() if len(v) >= 2
        ]
        self.itemnum = itemnum
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No eligible training histories")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        h = self.histories[index]
        # Usually supervise the recent window; random prefixes expose older data.
        end = len(h)
        if len(h) > 2 and np.random.random() < 0.35:
            end = np.random.randint(2, len(h) + 1)
        segment = h[max(0, end - self.maxlen - 1) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        seq[-(len(segment) - 1) :] = segment[:-1]
        pos[-(len(segment) - 1) :] = segment[1:]
        # All negative supervision is derived from training data alone.
        neg = np.random.randint(1, self.itemnum + 1, NEGATIVES)
        bad = np.isin(neg, h)
        if len(np.unique(h)) >= self.itemnum:
            raise ValueError("A training user has consumed every item")
        while bad.any():
            neg[bad] = np.random.randint(1, self.itemnum + 1, int(bad.sum()))
            bad = np.isin(neg, h)
        return seq, pos, neg


def worker_seed(worker_id):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s)
    random.seed(s)


class CausalResidual(nn.Module):
    def __init__(self, dim, dilation, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, 3, dilation=dilation)
        self.left_pad = 2 * dilation
        self.dropout = nn.Dropout(dropout)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def causal(self, conv, x):
        x = F.pad(x.transpose(1, 2), (self.left_pad, 0))
        return conv(x).transpose(1, 2)

    def forward(self, x, mask):
        y = self.causal(self.conv1, self.norm1(x) * mask)
        y = self.dropout(F.gelu(y)) * mask
        y = self.causal(self.conv2, self.norm2(y) * mask)
        return (x + self.gain * self.dropout(y)) * mask


class ConvRec(nn.Module):
    def __init__(self, itemnum, dim=96, maxlen=100, dropout=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        # Jointly trained target representations specialize independently of history.
        self.output_emb = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [CausalResidual(dim, d, dropout) for d in (1, 2, 4, 8, 1, 2, 4, 8)]
        )
        # Restore the attention block absent from the supplied ConvRec.
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, 2, dropout=0.2, batch_first=True
        )
        self.attention_dropout = nn.Dropout(0.2)
        # Initialize the jointly trained preference half-life to 20 positions.
        self.log_decay = nn.Parameter(
            torch.tensor(math.log(math.expm1(math.log(2.0) / 20.0)))
        )
        self.history_projection = nn.Linear(dim, dim)
        self.history_norm = nn.LayerNorm(dim)
        self.history_gate = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.history_gate.weight)
        nn.init.constant_(self.history_gate.bias, -2.0)
        self.final_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.normal_(self.output_emb.weight, std=0.1)
        nn.init.zeros_(self.bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.output_emb.weight[0].zero_()

    def forward(self, seq):
        mask = (seq != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        raw = self.item_emb(seq) * mask
        # Normalize causal exponential weights over nonpadding items only.
        positions = torch.arange(seq.size(1), device=seq.device)
        lag = positions[:, None] - positions[None, :]
        weights = torch.exp(
            -F.softplus(self.log_decay) * lag.clamp_min(0).to(raw.dtype)
        ).masked_fill(lag < 0, 0)
        pooled = (weights @ raw) / (weights @ mask).clamp_min(1e-8)
        pooled = pooled * mask
        x = self.input_dropout(raw) * mask
        for block in self.blocks:
            x = block(x, mask)

        length = seq.size(1)
        causal = torch.ones(
            length, length, dtype=torch.bool, device=seq.device
        ).tril()
        allowed = causal.unsqueeze(0) & (seq != 0).unsqueeze(1)
        # Only otherwise fully masked rows receive a safe diagonal entry.
        diagonal = torch.eye(length, dtype=torch.bool, device=seq.device)
        allowed = allowed | (~allowed.any(dim=-1, keepdim=True) & diagonal)
        attention_mask = (~allowed).repeat_interleave(
            self.attention.num_heads, dim=0
        )
        normalized = self.attention_norm(x) * mask
        attended, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=attention_mask, need_weights=False,
        )
        x = (x + self.attention_dropout(attended)) * mask
        p = self.history_norm(self.history_projection(pooled)) * mask
        g = torch.sigmoid(self.history_gate(torch.cat([x, p], dim=-1)))
        return self.final_norm(((1 - g) * x + g * p) * mask) * mask

    def training_loss(self, seq, pos, neg):
        feats = self(seq)
        positive = (feats * self.output_emb(pos)).sum(-1)
        positive = positive + self.bias(pos).squeeze(-1)
        negative = torch.einsum("bld,bnd->bln", feats, self.output_emb(neg))
        negative = negative + self.bias(neg).squeeze(-1).unsqueeze(1)
        logits = torch.cat([positive.unsqueeze(-1), negative], dim=-1)
        valid = pos != 0
        logits = logits[valid].float()
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=seq.device)
        return F.cross_entropy(logits, labels)

    def predict(self, seq, candidates):
        feats = self(seq)[:, -1]
        return (self.output_emb(candidates) * feats.unsqueeze(1)).sum(-1) + self.bias(
            candidates
        ).squeeze(-1)


def validation_examples(train, valid, itemnum, maxlen):
    # Local equivalent of the supplied baseline validation sampler.
    # Isolated RNG prevents validation from changing the training sequence.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Fewer than 99 available validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(pack_history(train[user], maxlen))
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError("No validation examples")
    return np.stack(sequences), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate(model, examples, device):
    model.eval()
    seqs, candidates = examples
    hits, ndcg = 0, 0.0
    for start in range(0, len(seqs), 128):
        seq = torch.as_tensor(seqs[start : start + 128], device=device)
        cand = torch.as_tensor(candidates[start : start + 128], device=device)
        scores = model.predict(seq, cand).float().cpu().numpy()
        # Stable descending ordering keeps the target-first tie convention.
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.argmax(order == 0, axis=1)
        hits += int((ranks < 10).sum())
        ndcg += float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0).sum())
    return hits / len(seqs), ndcg / len(seqs)


def check_embedding_roles(device):
    import io

    # Diagnostic initialization must not change subsequent training RNG states.
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        model = ConvRec(itemnum=8, dim=8, maxlen=6, dropout=0.0).to(device)
        model.eval()
        seq = torch.tensor([[0, 0, 1, 2, 3, 4]], device=device)
        candidates = torch.tensor([[1, 2, 3, 4]], device=device)
        assert model.item_emb.weight.data_ptr() != model.output_emb.weight.data_ptr()
        assert not torch.equal(model.item_emb.weight[1:], model.output_emb.weight[1:])
        assert torch.count_nonzero(model.item_emb.weight[0]).item() == 0
        assert torch.count_nonzero(model.output_emb.weight[0]).item() == 0
        with torch.no_grad():
            features = model(seq)
            changed = seq.clone()
            changed[:, 4:] = torch.tensor([7, 8], device=device)
            torch.testing.assert_close(
                features[:, :4], model(changed)[:, :4], rtol=1e-5, atol=1e-6
            )
            assert torch.isfinite(features).all().item()
            assert torch.count_nonzero(features[:, :2]).item() == 0
            empty = torch.zeros_like(seq)
            empty_features = model(empty)
            assert torch.isfinite(empty_features).all().item()
            assert torch.count_nonzero(empty_features).item() == 0
            assert torch.isfinite(model.predict(empty, candidates)).all().item()
            expected = model.predict(seq, candidates)

        # Both history and target tables must receive finite, nonzero gradients.
        pos = torch.tensor([[0, 0, 2, 3, 4, 5]], device=device)
        neg = torch.tensor([[6, 7, 8] * 10 + [6, 7]], device=device)
        loss = model.training_loss(seq, pos, neg)
        assert torch.isfinite(loss).item()
        loss.backward()
        for embedding in (model.item_emb, model.output_emb):
            gradient = embedding.weight.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all().item()
            assert torch.count_nonzero(gradient[1:]).item() > 0
            assert torch.count_nonzero(gradient[0]).item() == 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all().item()

        # Round-trip a real serialized checkpoint and require strict parity.
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = ConvRec(itemnum=8, dim=8, maxlen=6, dropout=0.0).to(device)
        restored.load_state_dict(
            torch.load(buffer, map_location=device, weights_only=True), strict=True
        )
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                expected, restored.predict(seq, candidates), rtol=0, atol=0
            )
    print("Embedding-role, causal, padding, and checkpoint diagnostics passed", flush=True)


def train_one(name, device, deadline):
    seed_all(42)
    train, valid, itemnum = load_data(name)
    dataset = Histories(train, itemnum, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        worker_init_fn=worker_seed,
        generator=generator,
        drop_last=False,
    )
    iterator = iter(loader)
    meta = dict(itemnum=itemnum, dim=DIM, maxlen=MAXLEN, dropout=DROPOUT)
    model = ConvRec(**meta).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=1e-4
    )
    examples = validation_examples(train, valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best_key = (-1.0, -1.0)
    best_epoch = 0
    # One selected checkpoint, never averaged with another checkpoint.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = 0.0
        steps_done = 0
        progress = (epoch - 1) / max(1, EPOCHS - 1)
        learning_rate = LR * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        for step in range(STEPS):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            seq, pos, neg = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(seq, pos, neg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses += loss.detach().item()
            steps_done += 1
            if time.monotonic() >= deadline:
                break
        hr, ndcg = evaluate(model, examples, device)
        key = (hr, ndcg)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        print(
            f"{name} epoch={epoch} loss={losses / max(1, steps_done):.5f} "
            f"HR@10={hr:.5f} NDCG@10={ndcg:.5f}",
            flush=True,
        )
        if epoch - best_epoch >= 25 or time.monotonic() >= deadline:
            break
    with open(os.path.join(WORKING, name + "_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    selected_hr, _ = evaluate(model, examples, device)
    del model, optimizer, iterator, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return selected_hr


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import pickle
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_pack_history = None


def _ensure_loaded():
    global _device, _pack_history
    if _models:
        return
    torch.set_num_threads(4)
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(_HERE, '..', 'train.py')
    spec = importlib.util.spec_from_file_location('convrec_training_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _pack_history = module.pack_history
    for name in _DATASETS:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        # Reconstruct the revised network with independent history and target tables.
        model = module.ConvRec(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'),
                           map_location=_device, weights_only=True)
        if 'output_emb.weight' not in state:
            raise RuntimeError('Checkpoint lacks output_emb for ' + name)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[name] = model
        _metas[name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _models:
        raise ValueError('Unknown dataset: ' + str(name))
    model = _models[name]
    seq = _pack_history(history, _metas[name]['maxlen'])
    seq = torch.as_tensor(seq, dtype=torch.long, device=_device).unsqueeze(0)
    cand = torch.as_tensor(candidates, dtype=torch.long, device=_device).unsqueeze(0)
    with torch.inference_mode():
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    overall_deadline = started + RUN_SECONDS
    check_embedding_roles(device)
    scores = []
    for index, name in enumerate(DATASETS):
        now = time.monotonic()
        # Equal shares of remaining wall time retain all four category models.
        deadline = now + max(1.0, overall_deadline - now) / (len(DATASETS) - index)
        scores.append(train_one(name, device, deadline))
    write_predict_script()
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()