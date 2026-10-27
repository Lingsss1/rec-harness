from __future__ import annotations

import json
import math
import os
import pickle
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
HERE = os.path.dirname(os.path.abspath(__file__))
WORKING = os.path.join(HERE, "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("SASREC_BATCH", "128"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "64"))
HIDDEN = int(os.environ.get("SASREC_HIDDEN", "96"))
BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
HEADS = int(os.environ.get("SASREC_HEADS", "2"))
DROPOUT = float(os.environ.get("SASREC_DROPOUT", "0.2"))
LR = float(os.environ.get("SASREC_LR", "0.0006"))
NEGATIVES = 1024
PATIENCE = 25
CATEGORY_SECONDS = 2400


def seed_everything(seed=42):
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


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    path = os.path.join(DATA_DIR, "dataset_stats.json")
    itemnum = max(max(items) for items in train.values() if items)
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if name in stats:
            itemnum = int(stats[name]["itemnum"])
    return train, valid, itemnum


def prediction_sequence(history, itemnum, maxlen):
    # The final position is always the query mask, during training and inference.
    context = list(history[-(maxlen - 1) :])
    result = [0] * maxlen
    result[-len(context) - 1 : -1] = context
    result[-1] = itemnum + 1
    return result


class MaskedHistories(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [h for h in train.values() if len(h) >= 2]
        self.itemnum = itemnum
        self.maxlen = maxlen

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        # Every training user is visited each epoch. Half the examples emphasize
        # the latest observed transition; the rest cover earlier prefixes.
        end = (
            len(history) - 1
            if random.random() < 0.5
            else random.randrange(1, len(history))
        )
        original = history[max(0, end - self.maxlen + 1) : end + 1]
        sequence = np.zeros(self.maxlen, dtype=np.int64)
        labels = np.zeros(self.maxlen, dtype=np.int64)
        offset = self.maxlen - len(original)
        sequence[offset:] = original
        for position in range(offset, self.maxlen):
            if position == self.maxlen - 1 or random.random() < 0.2:
                labels[position] = sequence[position]
                sequence[position] = self.itemnum + 1
        return torch.from_numpy(sequence), torch.from_numpy(labels)


def worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class MaskedSASRec(nn.Module):
    def __init__(
        self,
        itemnum,
        hidden_units=96,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        maxlen=64,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.input_norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        layer = nn.TransformerEncoderLayer(
            hidden_units,
            num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_blocks, enable_nested_tensor=False
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_units, hidden_units), nn.GELU(), nn.LayerNorm(hidden_units)
        )
        # This bias is fitted jointly with all other network parameters.
        self.item_bias = nn.Embedding(itemnum + 2, 1, padding_idx=0)
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.item_bias.weight.zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, sequence):
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        x = self.item_emb(sequence) + self.pos_emb(positions)[None]
        x = self.dropout(self.input_norm(x))
        x = self.encoder(x, src_key_padding_mask=sequence.eq(0))
        return self.output(x)

    def predict(self, sequence, candidates):
        query = self(sequence)[:, -1]
        return (self.item_emb(candidates) * query[:, None]).sum(-1) + self.item_bias(
            candidates
        ).squeeze(-1)


def make_validation(train, valid, itemnum, maxlen):
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
            raise ValueError("Official validation requires 99 unique negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(prediction_sequence(train[user], itemnum, maxlen))
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError("No eligible validation users")
    return torch.tensor(sequences), torch.tensor(candidates)


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    hits, ndcg = 0.0, 0.0
    for start in range(0, len(sequences), 64):
        scores = model.predict(
            sequences[start : start + 64].to(device),
            candidates[start : start + 64].to(device),
        )
        # Stable descending ordering matches target-first candidate tie handling.
        ordering = torch.argsort(scores, dim=1, descending=True, stable=True)
        ranks = ordering.eq(0).long().argmax(dim=1)
        hits += ranks.lt(10).float().sum().item()
        ndcg += (ranks.lt(10).float() / torch.log2(ranks.float() + 2)).sum().item()
    return hits / len(sequences), ndcg / len(sequences)


def train_one(name, device):
    seed_everything(42)
    train, valid, itemnum = load_dataset(name)
    meta = dict(
        itemnum=itemnum,
        hidden_units=HIDDEN,
        num_blocks=BLOCKS,
        num_heads=HEADS,
        dropout_rate=DROPOUT,
        maxlen=MAXLEN,
    )
    model = MaskedSASRec(**meta).to(device)
    dataset = MaskedHistories(train, itemnum, MAXLEN)
    if not len(dataset):
        raise ValueError("No training histories with two or more items")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        generator=torch.Generator().manual_seed(42),
    )
    validation = make_validation(train, valid, itemnum, MAXLEN)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=0.01
    )
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best, best_epoch = (-1.0, -1.0), 0
    start_time = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for sequence, labels in loader:
            sequence = sequence.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features = model(sequence)
            active = labels.ne(0)
            query = features[active]
            targets = labels[active]
            # Uniform negatives match the benchmark's candidate distribution.
            negatives = torch.randint(1, itemnum + 1, (NEGATIVES,), device=device)
            positive_logits = (query * model.item_emb(targets)).sum(-1)
            positive_logits = positive_logits + model.item_bias(targets).squeeze(-1)
            negative_logits = query @ model.item_emb(negatives).T
            negative_logits = negative_logits + model.item_bias(negatives).view(1, -1)
            negative_logits = negative_logits.masked_fill(
                targets[:, None].eq(negatives[None]), -1e4
            )
            logits = torch.cat((positive_logits[:, None], negative_logits), dim=1)
            per_position = nn.functional.cross_entropy(
                logits,
                torch.zeros(len(query), device=device, dtype=torch.long),
                reduction="none",
            )
            # Give the prediction-aligned final mask half the objective weight.
            last_mask = torch.zeros_like(active)
            last_mask[:, -1] = True
            is_last = last_mask[active]
            final_loss = per_position[is_last].mean()
            if (~is_last).any():
                loss = 0.5 * final_loss + 0.5 * per_position[~is_last].mean()
            else:
                loss = final_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach().item()
            steps += 1
        hr, ndcg = evaluate(model, validation, device)
        if (hr, ndcg) > best:
            best, best_epoch = (hr, ndcg), epoch
            torch.save(model.state_dict(), checkpoint)
        print(
            f"{name} epoch={epoch} loss={total_loss/max(steps,1):.4f} "
            f"HR@10={hr:.4f} NDCG@10={ndcg:.4f}",
            flush=True,
        )
        if (
            epoch - best_epoch >= PATIENCE
            or time.monotonic() - start_time >= CATEGORY_SECONDS
        ):
            break
    with open(os.path.join(WORKING, name + "_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    # Report the selected checkpoint's actual validation metric.
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    score, _ = evaluate(model, validation, device)
    del loader, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_module = None
_device = None


def _load(dataset_name):
    global _module, _device
    if dataset_name in _models:
        return
    if _module is None:
        spec = importlib.util.spec_from_file_location(
            'masked_sasrec_architecture', os.path.join(_HERE, '..', 'train.py'))
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _module.MaskedSASRec(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    _load(name)
    meta = _metas[name]
    sequence = _module.prediction_sequence(history, meta['itemnum'], meta['maxlen'])
    sequence = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor([candidates], dtype=torch.long, device=_device)
    with torch.inference_mode():
        return _models[name].predict(sequence, items)[0].cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(script)


def main():
    torch.set_num_threads(4)
    seed_everything(42)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WORKING, exist_ok=True)
    scores = [train_one(name, device) for name in DATASETS]
    write_predict_script()
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
