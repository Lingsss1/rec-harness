from __future__ import annotations

import json
import math
import os
import random
import time
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
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
BATCH_SIZE = int(os.environ.get("SASREC_BATCH", "128"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
DIM = int(os.environ.get("SIM_HIDDEN", "96"))
NEGATIVES = int(os.environ.get("SIM_NEGATIVES", "127"))
LR = float(os.environ.get("SIM_LR", "0.001"))
PATIENCE = int(os.environ.get("SIM_PATIENCE", "35"))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        itemnum = int(stats[name]["itemnum"])
    else:
        itemnum = max(item for history in train.values() for item in history)
    return train, valid, itemnum


class PrefixDataset(Dataset):
    """Random training-only prefixes, with extra emphasis on recent endpoints."""

    def __init__(self, train, itemnum, maxlen, examples, negatives):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.seen = [set(h.tolist()) for h in self.histories]
        if not self.histories:
            raise ValueError("No users with at least two training interactions")
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.examples = examples
        self.negatives = negatives

    def __len__(self):
        return self.examples

    def __getitem__(self, index):
        user = np.random.randint(len(self.histories))
        history = self.histories[user]
        if np.random.random() < 0.5:
            end = len(history) - 1
        else:
            end = np.random.randint(1, len(history))
        prefix = history[max(0, end - self.maxlen) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-len(prefix) :] = prefix
        target = int(history[end])
        seen = self.seen[user]
        # Exclude observed training positives, without consulting validation.
        # Replacement is permitted for training negatives only.
        if len(seen) >= self.itemnum:
            raise ValueError("No unseen training-negative items available")
        neg = np.random.randint(1, self.itemnum + 1, self.negatives)
        bad = np.asarray([int(x) in seen for x in neg], dtype=bool)
        while bad.any():
            neg[bad] = np.random.randint(1, self.itemnum + 1, int(bad.sum()))
            bad = np.asarray([int(x) in seen for x in neg], dtype=bool)
        candidates = np.concatenate((np.asarray([target], dtype=np.int64), neg))
        return seq, candidates


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class CausalHistoryBlock(nn.Module):
    """Contextualize history while keeping padded attention rows finite."""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, history, mask):
        batch, length, _ = history.shape
        causal = torch.ones(
            length, length, dtype=torch.bool, device=history.device
        ).triu(1)
        blocked = causal.unsqueeze(0) | ~mask.unsqueeze(1)
        # Padded queries may attend to themselves, preventing fully masked rows.
        # Valid queries still attend only to valid, causally preceding keys.
        diagonal = torch.eye(length, dtype=torch.bool, device=history.device)
        blocked = blocked & ~(~mask.unsqueeze(2) & diagonal.unsqueeze(0))
        blocked = blocked.repeat_interleave(self.num_heads, dim=0)
        normalized = self.attention_norm(history)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=blocked, need_weights=False,
        )
        history = (history + self.dropout(attended)).masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )
        history = (history + self.dropout(
            self.feedforward(self.feedforward_norm(history))
        )).masked_fill(~mask.unsqueeze(-1), 0.0)
        return history


class ItemAttention(nn.Module):
    """One jointly trained, candidate-conditioned item-similarity network."""

    def __init__(self, itemnum, dim=96, maxlen=128, dropout=0.2,
                 num_blocks=2, num_heads=2):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.dim = dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.encoder_dropout = dropout
        self.position_embedding = nn.Embedding(maxlen, dim)
        self.history_blocks = nn.ModuleList([
            CausalHistoryBlock(dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.history_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.position_embedding.weight, std=0.05)
        self.history_embedding = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.target_embedding = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.history_key = nn.Linear(dim, dim, bias=False)
        self.target_query = nn.Linear(dim, dim, bias=False)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.log_scale = nn.Parameter(torch.tensor(math.log(12.0)))
        self.log_attention_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.recency_raw = nn.Parameter(torch.tensor(-4.0))
        for embedding in (self.history_embedding, self.target_embedding):
            nn.init.normal_(embedding.weight, std=0.05)
            with torch.no_grad():
                embedding.weight[0].zero_()
        nn.init.zeros_(self.item_bias.weight)
        nn.init.orthogonal_(self.history_key.weight)
        nn.init.orthogonal_(self.target_query.weight)

    def forward(self, seq, candidates):
        mask = seq.ne(0)
        positions = torch.arange(seq.shape[1], device=seq.device)
        history = self.dropout(
            self.history_embedding(seq) + self.position_embedding(positions)
        ).masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.history_blocks:
            history = block(history, mask)
        # Both candidate-conditioned keys and pooled values use contextual history.
        history = nn.functional.normalize(self.history_norm(history), dim=-1)
        history = history.masked_fill(~mask.unsqueeze(-1), 0.0)
        targets = nn.functional.normalize(self.target_embedding(candidates), dim=-1)
        keys = nn.functional.normalize(self.history_key(history), dim=-1)
        queries = nn.functional.normalize(self.target_query(targets), dim=-1)
        attention = torch.bmm(queries, keys.transpose(1, 2))
        attention = attention * self.log_attention_scale.exp().clamp(max=20.0)
        distance = torch.arange(
            seq.shape[1] - 1, -1, -1, device=seq.device, dtype=history.dtype
        )
        decay = nn.functional.softplus(self.recency_raw)
        attention = attention - decay * distance.view(1, 1, -1)
        attention = attention.masked_fill(~mask.unsqueeze(1), -1e4)
        weights = torch.softmax(attention, dim=-1)
        weights = weights * mask.unsqueeze(1)
        # Dropout acts on the shared history values, retaining coherent contexts.
        context = torch.bmm(weights, self.dropout(history))
        context = nn.functional.normalize(context, dim=-1, eps=1e-6)
        compatibility = (context * targets).sum(-1)
        return self.log_scale.exp().clamp(max=40.0) * compatibility + self.item_bias(
            candidates
        ).squeeze(-1)

    def predict(self, seq, candidates):
        return self.forward(seq, candidates)


def prepare_validation(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        seq = np.zeros(maxlen, dtype=np.int64)
        history = train[user][-maxlen:]
        if history:
            seq[-len(history) :] = history
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Validation requires 99 unique unseen negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError("No validation users available")
    return (
        torch.from_numpy(np.stack(sequences)),
        torch.tensor(candidates, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate(model, validation, device):
    model.eval()
    seqs, candidates = validation
    hits, ndcg, count = 0.0, 0.0, 0
    for start in range(0, len(seqs), 64):
        scores = model.predict(
            seqs[start : start + 64].to(device),
            candidates[start : start + 64].to(device),
        ).float()
        # The positive is first, matching stable descending-sort tie handling.
        ranks = (scores[:, 1:] > scores[:, :1]).sum(dim=1)
        hit = ranks < 10
        hits += hit.float().sum().item()
        ndcg += (hit.float() / torch.log2(ranks.float() + 2)).sum().item()
        count += len(ranks)
    return hits / count, ndcg / count


def save_checkpoint(path, model, config, epoch, hr, ndcg):
    torch.save(
        {
            "config": dict(
                config,
                dim=model.dim,
                num_blocks=model.num_blocks,
                num_heads=model.num_heads,
                dropout=model.encoder_dropout,
            ),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "epoch": epoch,
            "validation_hr10": hr,
            "validation_ndcg10": ndcg,
        },
        path,
    )


def train_category(name, device):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    # Preserve the training protocol; isolate the contextual history encoder.
    config = dict(
        itemnum=itemnum, dim=DIM, maxlen=MAXLEN, dropout=0.2,
        num_blocks=2, num_heads=2,
    )
    model = ItemAttention(**config).to(device)
    validation = prepare_validation(train, valid, itemnum, MAXLEN)
    dataset = PrefixDataset(train, itemnum, MAXLEN, STEPS * BATCH_SIZE, NEGATIVES)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
    )
    # Embedding norms are controlled directly by normalized scoring.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-5
    )
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best_key = (-1.0, -1.0)
    best_epoch = 0
    started = time.monotonic()
    # Four sequential categories stay below the nine-hour run limit.
    category_time_limit = 6900.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq, candidates in loader:
            seq = seq.to(device, non_blocking=True)
            candidates = candidates.to(device, non_blocking=True)
            scores = model(seq, candidates)
            target = torch.zeros(len(seq), dtype=torch.long, device=device)
            loss = nn.functional.cross_entropy(scores, target)
            # A small train-only embedding penalty discourages uncontrolled norms.
            penalty = model.history_embedding(seq).square().mean()
            penalty = penalty + model.target_embedding(candidates).square().mean()
            loss = loss + 1e-4 * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        due = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS
        out_of_time = time.monotonic() - started >= category_time_limit
        if due or out_of_time:
            hr, ndcg = evaluate(model, validation, device)
            key = (hr, ndcg)
            if key > best_key:
                best_key = key
                best_epoch = epoch
                save_checkpoint(checkpoint, model, config, epoch, hr, ndcg)
            print(
                f"{name} epoch={epoch} loss={total_loss / len(loader):.5f} "
                f"val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}",
                flush=True,
            )
            if epoch - best_epoch >= PATIENCE or out_of_time:
                break
        else:
            print(
                f"{name} epoch={epoch} loss={total_loss / len(loader):.5f}", flush=True
            )
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    hr, _ = evaluate(model, validation, device)
    del loader, dataset, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return hr


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
_MODEL_CLASS = None


def _get_model(dataset):
    global _MODEL_CLASS
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if dataset not in _MODELS:
        if _MODEL_CLASS is None:
            spec = importlib.util.spec_from_file_location(
                'trained_item_attention', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODEL_CLASS = module.ItemAttention
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                           map_location=_DEVICE, weights_only=True)
        model = _MODEL_CLASS(**state['config']).to(_DEVICE)
        # Reconstruct exactly one contextual network from its saved configuration.
        model.load_state_dict(state['state_dict'], strict=True)
        model.eval()
        _MODELS[dataset] = model
    return _MODELS[dataset]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _get_model(dataset)
    hist = list(history[-model.maxlen:])
    seq = [0] * (model.maxlen - len(hist)) + hist
    seq_tensor = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    return model.predict(seq_tensor, candidate_tensor)[0].float().cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as handle:
        handle.write(script)


def main():
    os.makedirs(WORKING, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_predict_script()
    scores = [train_category(name, device) for name in DATASETS]
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()