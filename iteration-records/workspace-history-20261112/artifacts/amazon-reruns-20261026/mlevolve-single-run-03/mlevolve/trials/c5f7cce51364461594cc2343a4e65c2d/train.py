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
NEGATIVES = 256
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
        # Complete training histories are exclusion metadata only. Computing
        # unique IDs here consumes no RNG draws and preserves prefix sampling.
        self.history_items = [
            torch.tensor(sorted(set(h)), dtype=torch.long) for h in self.histories
        ]
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
        # The final-query view retains all context available at inference.
        sequence = np.asarray(
            prediction_sequence(history[:end], self.itemnum, self.maxlen),
            dtype=np.int64,
        )
        labels = np.zeros(self.maxlen, dtype=np.int64)
        labels[-1] = history[end]
        # Preserve the reference corruption draws in a separate aligned view.
        auxiliary_sequence = np.zeros(self.maxlen, dtype=np.int64)
        auxiliary_labels = np.zeros(self.maxlen, dtype=np.int64)
        offset = self.maxlen - len(original)
        auxiliary_sequence[offset:] = original
        for position in range(offset, self.maxlen):
            if position == self.maxlen - 1 or random.random() < 0.2:
                auxiliary_labels[position] = auxiliary_sequence[position]
                auxiliary_sequence[position] = self.itemnum + 1
        # Only historical reconstruction contributes to the auxiliary loss.
        auxiliary_labels[-1] = 0
        return (
            torch.from_numpy(sequence), torch.from_numpy(labels),
            torch.from_numpy(auxiliary_sequence), torch.from_numpy(auxiliary_labels),
            self.history_items[index],
        )


def collate_histories(examples):
    # Keep encoder inputs unchanged; concatenate sparse exclusion metadata
    # with its originating batch rows instead of allocating a catalog mask.
    sequence, labels, auxiliary_sequence, auxiliary_labels, histories = zip(*examples)
    history_items = torch.cat(histories)
    history_rows = torch.repeat_interleave(
        torch.arange(len(examples), dtype=torch.long),
        torch.tensor([len(items) for items in histories], dtype=torch.long),
    )
    return (
        torch.stack(sequence), torch.stack(labels),
        torch.stack(auxiliary_sequence), torch.stack(auxiliary_labels),
        history_items, history_rows,
    )


def history_negative_mask(history_items, history_rows, negatives, batch_size, itemnum):
    # Composite row/item keys implement sparse membership. Only the resulting
    # batch-by-256 mask is dense, regardless of catalog size or history length.
    if history_items.numel() == 0:
        return torch.zeros(
            (batch_size, negatives.numel()), dtype=torch.bool, device=negatives.device
        )
    stride = itemnum + 1
    keys = (history_rows * stride + history_items).sort().values
    sampled_keys = (
        torch.arange(batch_size, device=negatives.device)[:, None] * stride
        + negatives[None]
    )
    positions = torch.searchsorted(keys, sampled_keys.reshape(-1)).view_as(sampled_keys)
    return (positions < keys.numel()) & keys[
        positions.clamp(max=keys.numel() - 1)
    ].eq(sampled_keys)


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
        # Both training views share this reference backbone and tied scoring head.

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _features(self, sequence):
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        x = self.item_emb(sequence) + self.pos_emb(positions)[None]
        x = self.dropout(self.input_norm(x))
        x = self.encoder(x, src_key_padding_mask=sequence.eq(0))
        return self.output(x)

    def forward(self, sequence):
        return self._features(sequence)

    def predict(self, sequence, candidates):
        query = self(sequence)[:, -1]
        return (self.item_emb(candidates) * query[:, None]).sum(-1) + self.item_bias(
            candidates
        ).squeeze(-1)


def top10_pairwise_loss(logits, eligible_negatives):
    # Change only clean final-query supervision. Rank the positive and eligible
    # negatives using detached scores; excluded draws never occupy rank slots.
    # Stable sorting preserves target-first ordering when scores tie.
    with torch.no_grad():
        ranking_logits = torch.cat((
            logits[:, :1].detach(),
            logits[:, 1:].detach().masked_fill(~eligible_negatives, -torch.inf),
        ), dim=1)
        ordering = torch.argsort(
            ranking_logits, dim=1, descending=True, stable=True
        )
        ranks = torch.empty_like(ordering)
        ranks.scatter_(
            1, ordering,
            torch.arange(logits.shape[1], device=logits.device)[None].expand_as(ordering),
        )
        discounts = torch.where(
            ranks < 10,
            torch.log2(ranks.to(logits.dtype) + 2.0).reciprocal(),
            torch.zeros_like(ranking_logits),
        )
        weights = (discounts[:, :1] - discounts[:, 1:]).abs()
        weights = weights.masked_fill(~eligible_negatives, 0.0)
    # Mask before softplus so excluded logits have exactly zero gradient.
    # Empty rows retain a differentiable zero contribution to the batch mean.
    differences = (logits[:, 1:] - logits[:, :1]).masked_fill(
        ~eligible_negatives, 0.0
    )
    pair_losses = nn.functional.softplus(differences)
    weight_sum = weights.sum(dim=1)
    denominator = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
    return ((pair_losses * weights).sum(dim=1) / denominator).mean()


def check_training_views():
    # Check prefix alignment, truncation and masked labels without changing RNGs.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(42)
        history = list(range(1, 13))
        dataset = MaskedHistories({1: history}, itemnum=12, maxlen=6)
        for _ in range(32):
            sequence, labels, auxiliary_sequence, auxiliary_labels, history_items = dataset[0]
            assert torch.equal(history_items, torch.tensor(history))
            end = history.index(int(labels[-1]))
            expected = torch.tensor(prediction_sequence(history[:end], 12, 6))
            assert torch.equal(sequence, expected)
            assert labels[:-1].eq(0).all()
            assert auxiliary_sequence[-1].item() == 13
            assert auxiliary_labels[-1].item() == 0
            active = auxiliary_labels.ne(0)
            assert auxiliary_sequence[active].eq(13).all()
            assert torch.equal(auxiliary_labels[active], sequence[active])
            assert torch.equal(auxiliary_sequence[~active], sequence[~active])
        # Verify full-history exclusions, repeated draws, batch-row ownership,
        # and auxiliary query order without consuming training RNG state.
        probe = MaskedHistories(
            {1: [1, 2, 1, 3, 4], 2: [2, 5, 6]}, itemnum=6, maxlen=3
        )
        batch = collate_histories([probe[0], probe[1]])
        history_items, history_rows = batch[-2:]
        assert torch.equal(history_items, torch.tensor([1, 2, 3, 4, 2, 5, 6]))
        assert torch.equal(history_rows, torch.tensor([0, 0, 0, 0, 1, 1, 1]))
        negatives = torch.tensor([1, 2, 3, 4]).repeat(64)
        excluded = history_negative_mask(history_items, history_rows, negatives, 2, 6)
        expected = torch.tensor([
            [True, True, True, True], [False, True, False, False]
        ]).repeat(1, 64)
        assert torch.equal(excluded, expected)
        active = torch.tensor([[False, True, False], [True, False, True]])
        query_rows = torch.cat((torch.arange(2), active.nonzero(as_tuple=True)[0]))
        assert torch.equal(query_rows, torch.tensor([0, 1, 0, 1, 1]))
        assert torch.equal(excluded[query_rows], expected[query_rows])
        empty_mask = history_negative_mask(
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
            negatives, 2, 6,
        )
        assert not empty_mask.any()
        with torch.random.fork_rng(devices=[]):
            # Match the training sentinel and temperature, including a row
            # with every sampled negative excluded. Positives stay unmasked.
            clean = torch.linspace(-1.0, 1.0, 2 * (NEGATIVES + 1)).reshape(2, -1)
            corrupted = clean.flip(1).clone().requires_grad_(True)
            clean_logits = torch.cat((
                clean[:, :1], clean[:, 1:].masked_fill(excluded, -1e4)
            ), dim=1)
            corrupted_logits = torch.cat((
                corrupted[:, :1], corrupted[:, 1:].masked_fill(excluded, -1e4)
            ), dim=1)
            supervised = nn.functional.cross_entropy(
                corrupted_logits, torch.zeros(2, dtype=torch.long), reduction="none"
            )
            assert supervised[0].item() == 0.0
            consistency = nn.functional.kl_div(
                nn.functional.log_softmax(corrupted_logits / 2.0, dim=1),
                nn.functional.softmax(clean_logits.detach() / 2.0, dim=1),
                reduction="batchmean",
            )
            checked_loss = supervised.mean() + 0.1 * 2.0 ** 2 * consistency
            assert torch.isfinite(checked_loss)
            checked_loss.backward()
            assert torch.isfinite(corrupted.grad).all()
            assert corrupted.grad[:, 1:][excluded].eq(0).all()
            # Deterministic probes consume no RNG draws. Include a history
            # exclusion, a target collision, and a row with no eligible draws.
            probe_negatives = torch.tensor([2, 3, 4, 5])
            probe_targets = torch.tensor([1, 5])
            probe_history = torch.tensor([
                [False, False, True, False], [True, True, True, True]
            ])
            eligible = ~(
                probe_history | probe_targets[:, None].eq(probe_negatives[None])
            )
            ranking_probe = torch.tensor([
                [0.0, 2.0, -1.0, 100.0, -2.0],
                [0.0, 100.0, -100.0, 50.0, 100.0],
            ], requires_grad=True)
            ranking_loss = top10_pairwise_loss(ranking_probe, eligible)
            assert torch.isfinite(ranking_loss)
            ranking_loss.backward()
            assert torch.isfinite(ranking_probe.grad).all()
            assert ranking_probe.grad[:, 1:][~eligible].eq(0).all()
            assert ranking_probe.grad[1].eq(0).all()
            # Gradient descent lowers the incorrectly high eligible negative
            # and raises the positive. Excluded high scores cannot change ranks.
            assert ranking_probe.grad[0, 1].item() > 0.0
            assert ranking_probe.grad[0, 0].item() < 0.0
            changed_exclusions = ranking_probe.detach().clone()
            changed_exclusions[:, 1:].masked_fill_(~eligible, -1000.0)
            torch.testing.assert_close(
                top10_pairwise_loss(changed_exclusions, eligible),
                ranking_loss.detach(), rtol=0.0, atol=0.0,
            )
            empty_probe = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
            empty_loss = top10_pairwise_loss(
                empty_probe, torch.zeros((1, 2), dtype=torch.bool)
            )
            assert torch.isfinite(empty_loss) and empty_loss.item() == 0.0
            empty_loss.backward()
            assert empty_probe.grad.eq(0).all()
            # A positive below rank 10 receives weight only from negatives
            # inside the top 10; lower negatives have zero discount difference.
            boundary_probe = torch.tensor(
                [[0.0] + list(range(12, 0, -1)) + [-1.0]],
                requires_grad=True,
            )
            boundary_loss = top10_pairwise_loss(
                boundary_probe, torch.ones((1, 13), dtype=torch.bool)
            )
            boundary_loss.backward()
            assert boundary_probe.grad[0, 1:11].gt(0).all()
            assert boundary_probe.grad[0, 11:].eq(0).all()
            model = MaskedSASRec(
                12, hidden_units=8, num_blocks=1, num_heads=2,
                dropout_rate=0.0, maxlen=6,
            ).eval()
            empty = torch.tensor([prediction_sequence([], 12, 6)])
            candidates = torch.tensor([[1, 2, 3]])
            with torch.no_grad():
                assert torch.isfinite(model.predict(empty, candidates)).all()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    print(
        "Prediction-view alignment, auxiliary masking, sparse history exclusions, "
        "and finite-loss checks passed", flush=True
    )


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
        collate_fn=collate_histories,
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
        for (
            sequence, labels, auxiliary_sequence, auxiliary_labels,
            history_items, history_rows,
        ) in loader:
            sequence = sequence.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            auxiliary_sequence = auxiliary_sequence.to(device, non_blocking=True)
            auxiliary_labels = auxiliary_labels.to(device, non_blocking=True)
            history_items = history_items.to(device, non_blocking=True)
            history_rows = history_rows.to(device, non_blocking=True)
            active = auxiliary_labels.ne(0)
            batch_size = sequence.shape[0]
            # Boolean indexing of reconstruction features is row-major; use
            # the identical ordering to map each query back to its user.
            query_rows = torch.cat((
                torch.arange(batch_size, device=device),
                active.nonzero(as_tuple=True)[0],
            ))
            # Retain both final-query features from the shared forward pass.
            # The existing supervised prediction and reconstruction losses remain.
            if active.any():
                features = model(torch.cat((sequence, auxiliary_sequence), dim=0))
                clean_query = features[:batch_size, -1]
                auxiliary_query = features[batch_size:, -1]
                query = torch.cat(
                    (clean_query, features[batch_size:][active]), dim=0
                )
                targets = torch.cat((labels[:, -1], auxiliary_labels[active]), dim=0)
            else:
                features = model(sequence)
                query = features[:, -1]
                targets = labels[:, -1]
            # Uniform negatives match the benchmark's candidate distribution.
            negatives = torch.randint(1, itemnum + 1, (NEGATIVES,), device=device)
            # Preserve the existing 256 shared uniform draws. Exclude every
            # known training item for this user, including items outside the
            # sampled prefix, without passing metadata into the network.
            excluded_negatives = history_negative_mask(
                history_items, history_rows, negatives, batch_size, itemnum
            )
            positive_logits = (query * model.item_emb(targets)).sum(-1)
            positive_logits = positive_logits + model.item_bias(targets).squeeze(-1)
            negative_logits = query @ model.item_emb(negatives).T
            negative_logits = negative_logits + model.item_bias(negatives).view(1, -1)
            negative_logits = negative_logits.masked_fill(
                excluded_negatives[query_rows]
                | targets[:, None].eq(negatives[None]), -1e4
            )
            logits = torch.cat((positive_logits[:, None], negative_logits), dim=1)
            per_position = nn.functional.cross_entropy(
                logits,
                torch.zeros(len(query), device=device, dtype=torch.long),
                reduction="none",
            )
            # Replace only clean final-query cross-entropy. Keep the shared
            # 256 draws, auxiliary CE, consistency, and supervised coefficients.
            # Remove history exclusions and target collisions before ranking.
            eligible_negatives = ~(
                excluded_negatives
                | labels[:, -1, None].eq(negatives[None])
            )
            final_loss = top10_pairwise_loss(
                logits[:batch_size], eligible_negatives
            )
            if len(per_position) > batch_size:
                loss = 0.5 * final_loss + 0.5 * per_position[batch_size:].mean()
            else:
                loss = final_loss
            # Align corrupted final-query preferences to the detached clean view.
            # Only examples with historical masks contribute to consistency.
            consistency_rows = active.any(dim=1)
            if consistency_rows.any():
                consistency_query = auxiliary_query[consistency_rows]
                consistency_targets = labels[consistency_rows, -1]
                consistency_positive = (
                    consistency_query * model.item_emb(consistency_targets)
                ).sum(-1)
                consistency_positive = consistency_positive + model.item_bias(
                    consistency_targets
                ).squeeze(-1)
                consistency_negative = consistency_query @ model.item_emb(negatives).T
                consistency_negative = consistency_negative + model.item_bias(
                    negatives
                ).view(1, -1)
                consistency_negative = consistency_negative.masked_fill(
                    excluded_negatives[consistency_rows]
                    | consistency_targets[:, None].eq(negatives[None]), -1e4
                )
                consistency_logits = torch.cat(
                    (consistency_positive[:, None], consistency_negative), dim=1
                )
                # Both distributions use the same positive, 256 sampled negatives,
                # tied embeddings, jointly trained bias, and identical history
                # exclusions plus target-collision masking. The clean logits
                # above already carry the same per-user negative mask.
                temperature = 2.0
                clean_probabilities = nn.functional.softmax(
                    logits[:batch_size][consistency_rows].detach() / temperature,
                    dim=1,
                )
                consistency_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(
                        consistency_logits / temperature, dim=1
                    ),
                    clean_probabilities,
                    reduction="batchmean",
                )
                loss = loss + 0.1 * temperature ** 2 * consistency_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: nonfinite training loss at epoch {epoch}")
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
        pickle.dump({"constructor": meta}, f)
    # Report the selected checkpoint's actual validation metric.
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    score, _ = evaluate(model, validation, device)
    # Exercise the actual exported entry point against the selected checkpoint.
    # This reload is only a parity check; predictions use one selected model.
    import importlib.util
    with torch.random.fork_rng(
        devices=[device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    ):
        spec = importlib.util.spec_from_file_location(
            "export_parity_predictor", os.path.join(WORKING, "predict.py")
        )
        exported = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exported)
        users = [u for u in valid if u in train and valid[u]][:3]
        with torch.inference_mode():
            for user in users + [None]:
                history = train[user] if user is not None else []
                candidates = list(range(1, min(itemnum, 16) + 1))
                sequence = torch.tensor(
                    [prediction_sequence(history, itemnum, MAXLEN)], device=device
                )
                items = torch.tensor([candidates], device=device)
                expected = model.predict(sequence, items)[0].cpu()
                actual = torch.tensor(exported.predict(
                    user, history, candidates, dataset_name=name
                ), dtype=expected.dtype)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        exported._models.clear()
        del exported
    print(f"{name} selected_epoch={best_epoch} selected_HR@10={score:.4f} export_parity=passed", flush=True)
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
        metadata = pickle.load(f)
    meta = metadata['constructor']
    model = _module.MaskedSASRec(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
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
    check_training_views()
    write_predict_script()
    scores = [train_one(name, device) for name in DATASETS]
    score = float(np.mean(scores))
    for name, category_score in zip(DATASETS, scores):
        print(f"Selected-checkpoint local {name} HR@10={category_score:.4f}", flush=True)
    print(
        "Host evaluation is authoritative; the following macro HR@10 is measured "
        "locally from the four selected checkpoints.",
        flush=True,
    )
    print(f"Final Validation Score: {score}", flush=True)


if __name__ == "__main__":
    main()
