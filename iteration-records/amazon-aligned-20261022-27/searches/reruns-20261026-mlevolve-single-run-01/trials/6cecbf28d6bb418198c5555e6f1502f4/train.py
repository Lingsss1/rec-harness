from __future__ import annotations

import io
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


        # Accept legacy encoder arguments without constructing encoder parameters.
        self.num_blocks = 0
        self.num_heads = num_heads
        self.encoder_dropout = dropout
=======
=======
        # Accept legacy encoder arguments without constructing encoder parameters.
        self.num_blocks = 0
        self.num_heads = num_heads
        self.encoder_dropout = dropout


class ItemAttention(nn.Module):
    """One jointly trained, candidate-conditioned item-similarity network."""

    def __init__(self, itemnum, dim=96, maxlen=128, dropout=0.2,
                 num_blocks=2, num_heads=2):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.dim = dim
        # Accept legacy encoder arguments without constructing encoder parameters.
        self.num_blocks = 0
        self.num_heads = num_heads
        self.encoder_dropout = dropout
        self.history_embedding = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        # Both scoring paths update the same item representations.
        self.target_embedding = self.history_embedding
        self.history_key = nn.Linear(dim, dim, bias=False)
        self.target_query = nn.Linear(dim, dim, bias=False)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.log_scale = nn.Parameter(torch.tensor(math.log(12.0)))
        self.log_attention_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.recency_raw = nn.Parameter(torch.tensor(-4.0))
        # Initialize the shared table exactly once.
        nn.init.normal_(self.history_embedding.weight, std=0.05)
        with torch.no_grad():
            self.history_embedding.weight[0].zero_()
        nn.init.zeros_(self.item_bias.weight)
        nn.init.orthogonal_(self.history_key.weight)
        nn.init.orthogonal_(self.target_query.weight)

    def forward(self, seq, candidates):
        mask = seq.ne(0)
        # Keys and pooled values use direct embeddings from the shared item table.
        history = nn.functional.normalize(self.history_embedding(seq), dim=-1)
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
            "architecture": "tied_direct_item_attention_v1",
            "epoch": epoch,
            "validation_hr10": hr,
            "validation_ndcg10": ndcg,
        },
        path,
    )


def verify_tied_checkpoint():
    """Check the architecture and serialization without changing training RNG."""
    # torch.manual_seed also seeds CUDA; restore every affected RNG afterward.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(0)
        config = dict(
            itemnum=7, dim=96, maxlen=128, dropout=0.2,
            num_blocks=0, num_heads=2,
        )
        model = ItemAttention(**config).cpu().eval()
        assert model.history_embedding is model.target_embedding
        assert model.history_embedding.padding_idx == 0
        assert model.num_blocks == 0
        encoder_names = ("position_embedding", "history_blocks", "history_norm")
        assert all(not hasattr(model, name) for name in encoder_names)
        assert not any(
            key.startswith(encoder_names) for key in model.state_dict()
        )
        assert torch.count_nonzero(model.history_embedding.weight[0]).item() == 0
        assert torch.count_nonzero(model.item_bias.weight[0]).item() == 0
        parameters = list(model.parameters())
        assert len(parameters) == len({id(p) for p in parameters})
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-5
        )
        optimizer_parameters = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        assert len(optimizer_parameters) == len({id(p) for p in optimizer_parameters})
        assert sum(
            p is model.history_embedding.weight for p in optimizer_parameters
        ) == 1
        seq = torch.zeros((2, config["maxlen"]), dtype=torch.long)
        seq[0, -3:] = torch.tensor([1, 2, 3])
        # Leave the second history entirely padded to exercise safe attention.
        candidates = torch.tensor([[4, 5, 6], [6, 7, 1]], dtype=torch.long)
        scores = model(seq, candidates)
        assert torch.isfinite(scores).all().item()
        torch.testing.assert_close(
            scores[1], model.item_bias(candidates[1]).squeeze(-1), rtol=0, atol=0
        )
        padded_values = model.history_embedding(seq)[seq.eq(0)]
        assert torch.count_nonzero(padded_values).item() == 0
        penalty = model.history_embedding(seq).square().mean()
        penalty = penalty + model.target_embedding(candidates).square().mean()
        loss = nn.functional.cross_entropy(
            scores, torch.zeros(2, dtype=torch.long)
        ) + 1e-4 * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.count_nonzero(model.history_embedding.weight.grad[0]).item() == 0
        assert torch.count_nonzero(model.item_bias.weight.grad[0]).item() == 0
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        assert torch.count_nonzero(model.history_embedding.weight[0]).item() == 0
        assert torch.count_nonzero(model.item_bias.weight[0]).item() == 0
        with torch.inference_mode():
            expected = model.predict(seq, candidates)
        assert torch.isfinite(expected).all().item()
        with io.BytesIO() as buffer:
            save_checkpoint(buffer, model, config, 0, 0.0, 0.0)
            buffer.seek(0)
            saved = torch.load(buffer, map_location="cpu", weights_only=True)
        assert saved["architecture"] == "tied_direct_item_attention_v1"
        assert saved["config"]["num_blocks"] == 0
        restored = ItemAttention(**saved["config"]).cpu().eval()
        restored.load_state_dict(saved["state_dict"], strict=True)
        assert restored.history_embedding is restored.target_embedding
        assert restored.history_embedding.padding_idx == 0
        assert restored.num_blocks == 0
        assert all(not hasattr(restored, name) for name in encoder_names)
        assert not any(
            key.startswith(encoder_names) for key in restored.state_dict()
        )
        assert torch.count_nonzero(restored.history_embedding.weight[0]).item() == 0
        assert torch.count_nonzero(restored.item_bias.weight[0]).item() == 0
        with torch.inference_mode():
            actual = restored.predict(seq, candidates)
        assert torch.isfinite(actual).all().item()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def train_category(name, device):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    # Preserve the training protocol; isolate direct shared item embeddings.
    config = dict(
        itemnum=itemnum, dim=DIM, maxlen=MAXLEN, dropout=0.2,
        num_blocks=0, num_heads=2,
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
    assert saved["architecture"] == "tied_direct_item_attention_v1"
    assert saved["config"]["num_blocks"] == 0
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
        if state.get('architecture') != 'tied_direct_item_attention_v1':
            raise ValueError('Expected a tied direct-embedding ItemAttention checkpoint')
        if state['config'].get('num_blocks') != 0:
            raise ValueError('Direct-embedding checkpoints require num_blocks=0')
        model = _MODEL_CLASS(**state['config']).to(_DEVICE)
        # Reconstruct exactly one direct-embedding network per category.
        model.load_state_dict(state['state_dict'], strict=True)
        assert model.history_embedding is model.target_embedding
        assert model.history_embedding.padding_idx == 0
        assert torch.count_nonzero(model.history_embedding.weight[0]).item() == 0
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
    verify_tied_checkpoint()
    scores = [train_category(name, device) for name in DATASETS]
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()