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
BATCH_SIZE = int(os.environ.get("NAIS_BATCH", "256"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
DIM = int(os.environ.get("NAIS_DIM", "96"))
NEGATIVES = 16
PATIENCE = 24
SECONDS_PER_CATEGORY = float(os.environ.get("NAIS_SECONDS_PER_CATEGORY", "1800"))
SASREC_MAXLEN = int(os.environ.get("SASREC_EXPERT_MAXLEN", "200"))
_SEARCH_STARTED = None
_CATEGORY_INDEX = 0
_SIMILARITY_DEADLINE = float("inf")
_SEARCH_SECONDS = min(
    float(os.environ.get("REMAINING_GPU_SECONDS", "43200")),
    8.5 * 3600,
)


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


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats[name]["itemnum"]) if name in stats else 0
    else:
        itemnum = 0
    itemnum = max(
        itemnum,
        max((max(h) for h in train.values() if h), default=0),
        max((max(h) for h in valid.values() if h), default=0),
    )
    return train, valid, itemnum


class PurchaseDataset(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64)
            for h in train.values()
            if len(set(h)) >= 2 and len(set(h)) < itemnum
        ]
        self.known = [set(h.tolist()) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        t = np.random.randint(1, len(history))
        target = int(history[t])
        context = history[:t]
        if len(context) > self.maxlen:
            if np.random.random() < 0.5:
                context = context[-self.maxlen :]
            else:
                indices = np.sort(
                    np.random.choice(len(context), self.maxlen, replace=False)
                )
                context = context[indices]
        if len(context) > 1:
            keep = np.random.random(len(context)) >= 0.15
            if not keep.any():
                keep[np.random.randint(len(context))] = True
            context = context[keep]
        sequence = np.zeros(self.maxlen, dtype=np.int64)
        sequence[-len(context) :] = context
        negatives = []
        known = self.known[index]
        while len(negatives) < NEGATIVES:
            item = np.random.randint(1, self.itemnum + 1)
            if item not in known:
                negatives.append(item)
        return sequence, np.asarray([target] + negatives, dtype=np.int64)


def check_purchase_dataset():
    from unittest.mock import patch

    state = np.random.get_state()
    try:
        dataset = PurchaseDataset({1: [1, 2, 1, 3], 2: [4, 5]}, 10, 8)
        examples = []
        for index, t in [(0, 1), (0, 2), (0, 3), (1, 1)]:
            history = dataset.histories[index]
            with patch.object(
                np.random, "randint", side_effect=[t] + [10] * NEGATIVES
            ), patch.object(
                np.random, "random", side_effect=lambda size: np.ones(size)
            ):
                sequence, candidates = dataset[index]
            assert np.array_equal(sequence[-t:], history[:t])
            assert not sequence[:-t].any()
            assert candidates[0] == history[t]
            assert all(item not in dataset.known[index] for item in candidates[1:])
            assert sequence.shape == (8,)
            assert candidates.shape == (1 + NEGATIVES,)
            examples.append((sequence, candidates))
        assert examples[1][0][-2] == examples[1][1][0] == 1
        with patch.object(
            np.random, "randint", side_effect=[3, 0] + [10] * NEGATIVES
        ), patch.object(np.random, "random", side_effect=lambda size: np.zeros(size)):
            sequence, _ = dataset[0]
        assert np.count_nonzero(sequence) == 1
        assert sequence[-1] == dataset.histories[0][0]
    finally:
        np.random.set_state(state)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CausalContextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads=2, dropout=0.1, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * dim, dim),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, context, attention_mask, mask):
        normalized = self.attention_norm(context)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        context = context + self.dropout(attended)
        context = context.masked_fill(~mask.unsqueeze(-1), 0.0)
        context = context + self.dropout(
            self.feedforward(self.feedforward_norm(context))
        )
        return context.masked_fill(~mask.unsqueeze(-1), 0.0)


class AttentiveSimilarity(nn.Module):
    def __init__(self, itemnum, dim=96, maxlen=128, attention_dim=32):
        super().__init__()
        self.itemnum = itemnum
        self.dim = dim
        self.maxlen = maxlen
        self.context = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.target = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.history_key = nn.Linear(dim, attention_dim, bias=False)
        self.candidate_key = nn.Linear(dim, attention_dim, bias=False)
        self.attention_dim = attention_dim
        self.recency_bias = nn.Parameter(torch.zeros(8))
        self.dropout = nn.Dropout(0.1)
        nn.init.normal_(self.context.weight, std=0.08)
        nn.init.normal_(self.target.weight, std=0.08)
        nn.init.zeros_(self.bias.weight)
        with torch.no_grad():
            self.context.weight[0].zero_()
            self.target.weight[0].zero_()
        self.position = nn.Embedding(maxlen, dim)
        self.encoder_blocks = nn.ModuleList([CausalContextBlock(dim) for _ in range(2)])
        self.context_gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.position.weight, std=0.02)

        # Dedicated next-item readout from the final causal sequence state.
        # Constructed after all parent parameters to preserve their initialization.
        self.sequence_readout = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
        )
        self.sequence_gate = nn.Parameter(torch.zeros(()))

    def encode_context(self, history, original_context):
        mask = history.ne(0)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        context = original_context + self.position(positions)
        context = context.masked_fill(~mask.unsqueeze(-1), 0.0)
        length = history.size(1)
        future = torch.ones(
            length, length, dtype=torch.bool, device=history.device
        ).triu(1)
        blocked = future.unsqueeze(0) | ~mask.unsqueeze(1)
        self_only = ~torch.eye(length, dtype=torch.bool, device=history.device)
        blocked = torch.where(mask.unsqueeze(-1), blocked, self_only.unsqueeze(0))
        attention_mask = blocked.repeat_interleave(2, dim=0)
        for block in self.encoder_blocks:
            context = block(context, attention_mask, mask)
        return context

    def score_components(self, history, candidates):
        mask = history.ne(0)
        r = (mask.long().flip(-1).cumsum(-1).flip(-1) - 1).clamp_min(0)
        r = r.masked_fill(~mask, 0)
        bucket = torch.floor(torch.log2(1.0 + r.float())).long().clamp(max=7)
        original_context = self.dropout(self.context(history))
        encoded_context = self.encode_context(history, original_context)
        context = original_context + torch.tanh(self.context_gate) * encoded_context
        target = self.target(candidates)
        keys = torch.tanh(self.history_key(context))
        queries = torch.tanh(self.candidate_key(target))
        compatibility = torch.bmm(queries, keys.transpose(1, 2))
        compatibility = compatibility / math.sqrt(self.attention_dim)
        attention = torch.exp(
            3.0 * torch.tanh(compatibility)
            + 2.0 * torch.tanh(self.recency_bias[bucket]).unsqueeze(1)
        ) * mask.unsqueeze(1)
        pair_scores = torch.bmm(target, context.transpose(1, 2))
        normalizer = attention.sum(-1).clamp_min(1.0).pow(0.5)
        aggregate = (attention * pair_scores).sum(-1) / normalizer
        aggregate = aggregate + self.bias(candidates).squeeze(-1)

        # Gather the final real position, also supporting empty histories.
        positions = torch.arange(history.size(1), device=history.device)
        final_index = positions.unsqueeze(0).expand_as(history)
        final_index = final_index.masked_fill(~mask, -1).amax(-1)
        final_state = encoded_context[
            torch.arange(history.size(0), device=history.device),
            final_index.clamp_min(0),
        ]
        query = self.sequence_readout(final_state)
        query = query * mask.any(-1, keepdim=True)
        sequential = torch.bmm(target, query.unsqueeze(-1)).squeeze(-1)
        return aggregate, sequential

    def forward(self, history, candidates):
        aggregate, sequential = self.score_components(history, candidates)
        return aggregate + self.sequence_gate.tanh() * sequential

    def predict(self, history, candidates):
        return self(history, candidates)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return outputs.transpose(-1, -2) + inputs


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=50, maxlen=200,
                 num_blocks=2, num_heads=1, dropout_rate=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate)
            for _ in range(num_blocks)])
        self.forward_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)])
        self.forward_layers = nn.ModuleList([
            PointWiseFeedForward(hidden_units, dropout_rate)
            for _ in range(num_blocks)])
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def log2feats(self, seq):
        seqs = self.item_emb(seq) * math.sqrt(self.hidden_units)
        positions = torch.arange(seq.size(1), device=seq.device)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions).unsqueeze(0))
        timeline_mask = seq.eq(0)
        seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)
        attention_mask = torch.ones(
            seq.size(1), seq.size(1), dtype=torch.bool, device=seq.device).triu(1)
        for norm, attention, forward_norm, feedforward in zip(
                self.attention_layernorms, self.attention_layers,
                self.forward_layernorms, self.forward_layers):
            seqs = seqs.transpose(0, 1)
            query = norm(seqs)
            seqs = query + attention(query, seqs, seqs,
                                    attn_mask=attention_mask,
                                    need_weights=False)[0]
            seqs = feedforward(forward_norm(seqs.transpose(0, 1)))
            seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)
        return self.last_layernorm(seqs)

    def forward(self, seq, pos=None, neg=None):
        features = self.log2feats(seq)
        if pos is None:
            return features
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1]
        return torch.bmm(self.item_emb(candidates), features.unsqueeze(-1)).squeeze(-1)


def prepare_history(history, maxlen):
    # Compact padding and retain the chronological suffix for each expert.
    if history.ndim == 1:
        history = history.unsqueeze(0)
    result = history.new_zeros((history.size(0), maxlen))
    for index, row in enumerate(history):
        items = row[row.ne(0)][-maxlen:]
        if items.numel():
            result[index, -items.numel():] = items
    return result


def rank_fusion(similarity_scores, sasrec_scores, candidates):
    # Rank distinct IDs once. Duplicate occurrences share a rank and never
    # displace another item. Stable item-ID ordering resolves raw-score ties.
    if similarity_scores.shape != candidates.shape or sasrec_scores.shape != candidates.shape:
        raise ValueError("Score and candidate shapes must agree")
    if not torch.isfinite(similarity_scores).all() or not torch.isfinite(sasrec_scores).all():
        raise FloatingPointError("Nonfinite expert scores")
    fused = torch.empty_like(similarity_scores, dtype=torch.float32)
    denominator = max(candidates.size(1) - 1, 1)
    for row in range(candidates.size(0)):
        ids, inverse = torch.unique(candidates[row], sorted=True, return_inverse=True)
        first = torch.full((ids.numel(),), candidates.size(1),
                           dtype=torch.long, device=candidates.device)
        first.scatter_reduce_(0, inverse,
                              torch.arange(candidates.size(1), device=candidates.device),
                              reduce="amin", include_self=True)
        ranks = []
        for scores in (similarity_scores, sasrec_scores):
            order = torch.argsort(scores[row, first], descending=True, stable=True)
            unique_ranks = torch.empty_like(order)
            unique_ranks[order] = torch.arange(ids.numel(), device=order.device)
            ranks.append(unique_ranks[inverse].float())
        fused[row] = -0.5 * (ranks[0] + ranks[1]) / denominator
    return fused


class RankEnsemble(nn.Module):
    def __init__(self, similarity, sasrec, chunk_size=256):
        super().__init__()
        self.similarity = similarity
        self.sasrec = sasrec
        self.maxlen = max(similarity.maxlen, sasrec.maxlen)
        self.chunk_size = chunk_size

    def predict(self, history, candidates):
        if candidates.size(1) == 0:
            return torch.empty(candidates.shape, device=candidates.device)
        member_scores = []
        for expert in (self.similarity, self.sasrec):
            sequence = prepare_history(history, expert.maxlen)
            # Fusion always sees the complete candidate set, even when raw
            # expert scoring must be chunked for memory.
            member_scores.append(torch.cat([
                expert.predict(sequence, candidates[:, start:start + self.chunk_size])
                for start in range(0, candidates.size(1), self.chunk_size)
            ], dim=1))
        return rank_fusion(*member_scores, candidates)


class SASRecDataset(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [h for h in train.values()
                          if len(h) > 1 and len(set(h)) < itemnum]
        self.known = [set(h) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros_like(seq)
        next_item = history[-1]
        position = self.maxlen - 1
        for item in reversed(history[:-1]):
            seq[position] = item
            pos[position] = next_item
            sampled = np.random.randint(1, self.itemnum + 1)
            while sampled in self.known[index]:
                sampled = np.random.randint(1, self.itemnum + 1)
            neg[position] = sampled
            next_item = item
            position -= 1
            if position < 0:
                break
        return seq, pos, neg


def load_expert(name, kind, device):
    checkpoint = torch.load(os.path.join(WORKING, name + "_" + kind + ".pt"),
                            map_location="cpu", weights_only=True)
    expert_class = AttentiveSimilarity if kind == "model" else SASRec
    model = expert_class(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def check_attentive_similarity():
    import io

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        config = dict(itemnum=8, dim=96, maxlen=12, attention_dim=32)
        model = AttentiveSimilarity(**config).eval()
        history = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 2, 1, 3],
                [1, 2, 3, 4, 5, 6],
            ]
        )
        candidates = torch.tensor([[2, 3, 4]]).expand(3, -1)
        mask = history.ne(0)
        with torch.no_grad():
            aggregate, sequential = model.score_components(history, candidates)
            assert model.sequence_gate.item() == 0.0
            torch.testing.assert_close(
                model.predict(history, candidates), aggregate, rtol=0, atol=0
            )
            model.sequence_gate.fill_(0.5)
            torch.testing.assert_close(
                model.predict(history, candidates),
                aggregate + math.tanh(0.5) * sequential,
            )
            model.context_gate.fill_(0.5)
            model.bias.weight[2:].fill_(0.1)
            encoded = model.encode_context(history, model.context(history))
            assert torch.count_nonzero(encoded[~mask]).item() == 0
            scores = model.predict(history, candidates)
            assert torch.isfinite(scores).all()
            torch.testing.assert_close(
                scores[0], model.bias(candidates[0]).squeeze(-1), rtol=0, atol=0
            )
            model.context.weight[0].fill_(100.0)
            torch.testing.assert_close(model.predict(history, candidates), scores)
            compact = torch.tensor([[0, 0, 1, 2, 1, 3]])
            torch.testing.assert_close(
                model.predict(compact, candidates[1:2]), scores[1:2]
            )
            changed = history.clone()
            changed[:, 4:] = torch.tensor([7, 8])
            changed_encoded = model.encode_context(changed, model.context(changed))
            torch.testing.assert_close(encoded[:, :4], changed_encoded[:, :4])
            short_encoded = model.encode_context(
                history[:, :4], model.context(history[:, :4])
            )
            torch.testing.assert_close(encoded[:, :4], short_encoded)
        model.zero_grad(set_to_none=True)
        model(history, candidates).sum().backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        assert model.sequence_gate.grad is not None
        assert model.sequence_readout[1].weight.grad is not None
        with torch.no_grad():
            scores = model.predict(history, candidates)
            buffer = io.BytesIO()
            torch.save({"config": config, "state_dict": model.state_dict()}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
            restored = AttentiveSimilarity(**checkpoint["config"]).eval()
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            torch.testing.assert_close(
                restored.predict(history, candidates), scores, rtol=0, atol=0
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
        history = train[user][-maxlen:]
        seq = np.zeros(maxlen, dtype=np.int64)
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
        raise ValueError("No eligible validation users")
    return (
        torch.from_numpy(np.stack(sequences)).long(),
        torch.tensor(candidates, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate(model, evaluation, device):
    model.eval()
    sequences, candidates = evaluation
    hits = 0.0
    ndcg = 0.0
    # Match the exported predictor's single-user forward-pass shapes.
    for start in range(len(sequences)):
        scores = model.predict(
            sequences[start : start + 1].to(device),
            candidates[start : start + 1].to(device),
        )
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite validation scores")
        ranks = (scores[:, 1:] > scores[:, :1]).sum(1).cpu().numpy()
        hits += float((ranks < 10).sum())
        ndcg += float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0).sum())
    return hits / len(sequences), ndcg / len(sequences)


def _train_similarity(name, device):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    dataset = PurchaseDataset(train, itemnum, MAXLEN)
    if not len(dataset):
        raise ValueError("No eligible training histories for " + name)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.RandomSampler(
            dataset,
            replacement=True,
            num_samples=200 * BATCH_SIZE,
            generator=generator,
        ),
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    config = dict(itemnum=itemnum, dim=DIM, maxlen=MAXLEN, attention_dim=32)
    model = AttentiveSimilarity(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=1e-5, betas=(0.9, 0.98)
    )
    evaluation = make_validation(train, valid, itemnum, MAXLEN)
    criterion = nn.CrossEntropyLoss()
    # Validate immediately so the deadline always leaves a loadable selection.
    best = evaluate(model, evaluation, device)
    best_epoch = 0
    best_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        if time.monotonic() >= _SIMILARITY_DEADLINE:
            break
        model.train()
        total_loss = 0.0
        examples = 0
        optimizer_steps = 0
        for histories, candidates in loader:
            if time.monotonic() >= _SIMILARITY_DEADLINE:
                break
            histories = histories.to(device, non_blocking=True)
            candidates = candidates.to(device, non_blocking=True)
            scores = model(histories, candidates)
            labels = torch.zeros(len(histories), dtype=torch.long, device=device)
            loss = criterion(scores, labels)
            reg = (
                model.context(histories).square().sum(-1) * histories.ne(0)
            ).sum() / histories.ne(0).sum().clamp_min(1)
            reg = reg + model.target(candidates).square().sum(-1).mean()
            loss = loss + 1e-4 * reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach()) * len(histories)
            examples += len(histories)
        if not examples:
            break
        if (epoch == 1 or epoch % 2 == 0 or epoch == EPOCHS
                or time.monotonic() >= _SIMILARITY_DEADLINE):
            metric = evaluate(model, evaluation, device)
            if metric > best:
                best = metric
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(
                f"{name} epoch={epoch} loss={total_loss / examples:.5f} "
                f"optimizer_steps={optimizer_steps} examples={examples} "
                f"validation_HR@10={metric[0]:.5f} "
                f"validation_NDCG@10={metric[1]:.5f} "
                f"sequence_gate={model.sequence_gate.detach().tanh().item():.4f}",
                flush=True,
            )
        else:
            print(
                f"{name} epoch={epoch} loss={total_loss / examples:.5f} "
                f"optimizer_steps={optimizer_steps} examples={examples}",
                flush=True,
            )
        if epoch - best_epoch >= PATIENCE:
            break
        if (time.monotonic() - started >= SECONDS_PER_CATEGORY
                or time.monotonic() >= _SIMILARITY_DEADLINE):
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    hr, _ = evaluate(model, evaluation, device)
    torch.save(
        {"config": config, "state_dict": best_state},
        os.path.join(WORKING, name + "_model.pt"),
    )
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return hr


def _train_sasrec(name, device, train, valid, itemnum, deadline):
    # Reset initialization, sampling, dropout, and worker streams independently.
    seed_everything(42)
    config = dict(itemnum=itemnum,
                  hidden_units=int(os.environ.get("SASREC_HIDDEN_UNITS", "50")),
                  maxlen=SASREC_MAXLEN,
                  num_blocks=int(os.environ.get("SASREC_NUM_BLOCKS", "2")),
                  num_heads=int(os.environ.get("SASREC_NUM_HEADS", "1")),
                  dropout_rate=float(os.environ.get("SASREC_DROPOUT_RATE", "0.2")))
    model = SASRec(**config).to(device)
    for parameter in model.parameters():
        if parameter.ndim > 1:
            nn.init.xavier_normal_(parameter)
    dataset = SASRecDataset(train, itemnum, config["maxlen"])
    if not len(dataset):
        raise ValueError("No SASRec training histories for " + name)
    batch_size = int(os.environ.get("SASREC_BATCH_SIZE", "128"))
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(
            dataset, replacement=True,
            num_samples=max(1, len(dataset) // batch_size) * batch_size,
            generator=generator),
        num_workers=2, pin_memory=device.type == "cuda",
        persistent_workers=True, worker_init_fn=seed_worker,
        generator=generator)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(os.environ.get("SASREC_LR", "0.001")),
        betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()
    evaluation = make_validation(train, valid, itemnum, config["maxlen"])
    best = evaluate(model, evaluation, device)
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
    for epoch in range(1, EPOCHS + 1):
        if time.monotonic() >= deadline:
            break
        model.train()
        steps = 0
        for seq, pos, neg in loader:
            if time.monotonic() >= deadline:
                break
            seq, pos, neg = [value.to(device, non_blocking=True)
                             for value in (seq, pos, neg)]
            positive, negative = model(seq, pos, neg)
            active = pos.ne(0)
            loss = criterion(positive[active], torch.ones_like(positive[active]))
            loss = loss + criterion(negative[active], torch.zeros_like(negative[active]))
            l2 = float(os.environ.get("SASREC_L2_EMB", "0.0"))
            if l2:
                loss = loss + l2 * torch.norm(model.item_emb.weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        if not steps:
            break
        # Independent lexicographic HR@10/NDCG@10 checkpoint selection.
        metric = evaluate(model, evaluation, device)
        if metric > best:
            best = metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        print(f"{name} SASRec epoch={epoch} validation_HR@10={metric[0]:.5f} "
              f"validation_NDCG@10={metric[1]:.5f}", flush=True)
        if epoch - best_epoch >= PATIENCE:
            break
    torch.save({"config": config, "state_dict": best_state},
               os.path.join(WORKING, name + "_sasrec.pt"))
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def train_category(name, device):
    global _SEARCH_STARTED, _CATEGORY_INDEX, _SIMILARITY_DEADLINE
    if _SEARCH_STARTED is None:
        _SEARCH_STARTED = time.monotonic()
    now = time.monotonic()
    # Divide the remaining wall-clock allowance across remaining categories;
    # single-GPU elapsed time conservatively bounds GPU time. The caller must
    # supply REMAINING_GPU_SECONDS when earlier trials consumed shared budget.
    remaining_categories = len(DATASETS) - _CATEGORY_INDEX
    remaining = _SEARCH_SECONDS - (now - _SEARCH_STARTED)
    reserve = 120.0 + 60.0 * remaining_categories
    if remaining <= reserve:
        raise RuntimeError("Insufficient remaining budget for all expert validations/export")
    allowance = (remaining - reserve) / remaining_categories
    _SIMILARITY_DEADLINE = now + min(SECONDS_PER_CATEGORY, allowance / 2)
    category_deadline = now + allowance
    _train_similarity(name, device)
    train, valid, itemnum = load_data(name)
    _train_sasrec(name, device, train, valid, itemnum, category_deadline)
    # Strict reloads, rather than live training objects, define deployment.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        similarity = load_expert(name, "model", device)
        sasrec = load_expert(name, "sasrec", device)
        ensemble = RankEnsemble(similarity, sasrec).eval()
        evaluation = make_validation(train, valid, itemnum, ensemble.maxlen)
        member_metrics = []
        for expert in (similarity, sasrec):
            member_evaluation = (prepare_history(evaluation[0], expert.maxlen),
                                 evaluation[1])
            member_metrics.append(evaluate(expert, member_evaluation, device))
        metric = evaluate(ensemble, evaluation, device)
        print(f"{name} similarity_HR@10={member_metrics[0][0]:.5f} "
              f"SASRec_HR@10={member_metrics[1][0]:.5f} "
              f"ensemble_HR@10={metric[0]:.5f} "
              f"ensemble_NDCG@10={metric[1]:.5f}", flush=True)
        check_predictor(name, ensemble, evaluation)
    del similarity, sasrec, ensemble
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _CATEGORY_INDEX += 1
    return metric[0]


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific',
          'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(name):
    global _CLASS
    if name not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(name))
    if name in _MODELS:
        return _MODELS[name]
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'attentive_similarity_train', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module
    experts = []
    for suffix, expert_class in (('model', _CLASS.AttentiveSimilarity),
                                ('sasrec', _CLASS.SASRec)):
        checkpoint = torch.load(os.path.join(_HERE, name + '_' + suffix + '.pt'),
                                map_location='cpu', weights_only=True)
        expert = expert_class(**checkpoint['config']).to(_DEVICE)
        expert.load_state_dict(checkpoint['state_dict'], strict=True)
        experts.append(expert.eval())
    model = _CLASS.RankEnsemble(*experts).eval()
    _MODELS[name] = model
    return model


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    name = dataset_name if dataset_name is not None else _NAMES[0]
    model = _load(name)
    if not candidates:
        return []
    hist = list(history)[-model.maxlen:]
    seq = [0] * (model.maxlen - len(hist)) + hist
    sequence = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    scores = model.predict(sequence, candidate_tensor)[0]
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Nonfinite prediction scores')
    return scores.cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(script)


@torch.inference_mode()
def check_predictor(name, model, evaluation):
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        spec = importlib.util.spec_from_file_location(
            "attentive_similarity_predict_check", os.path.join(WORKING, "predict.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model.eval()
        sequences, candidates = evaluation
        device = next(model.parameters()).device
        # Deterministic diagnostics use no sampling and restore model chunk size.
        saved_chunk_size = model.chunk_size
        try:
            for index in range(min(3, len(sequences))):
                histories = [sequences[index][sequences[index].ne(0)].tolist(), []]
                choices = candidates[index].repeat(3).to(device)
                candidate_batch = choices.unsqueeze(0)
                for history in histories:
                    sequence = torch.tensor([history], dtype=torch.long, device=device)
                    model.chunk_size = 256
                    expected = model.predict(sequence, candidate_batch)[0]
                    assert torch.isfinite(expected).all()
                    torch.testing.assert_close(expected[:100], expected[100:200],
                                               rtol=0, atol=0)
                    torch.testing.assert_close(expected[:100], expected[200:300],
                                               rtol=0, atol=0)
                    actual = torch.tensor(module.predict(
                        index, history, choices.tolist(), dataset_name=name),
                        dtype=expected.dtype, device=device)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    # Compare identical deployed inputs. Changing candidate
                    # layout can perturb raw scores and therefore fused ranks.
                    permutation = torch.arange(choices.numel() - 1, -1, -1,
                                               device=device)
                    permuted_choices = choices[permutation]
                    permuted_expected = model.predict(
                        sequence, permuted_choices.unsqueeze(0))[0]
                    assert torch.isfinite(permuted_expected).all()
                    permuted_export = torch.tensor(module.predict(
                        index, history, permuted_choices.tolist(), dataset_name=name),
                        dtype=expected.dtype, device=device)
                    torch.testing.assert_close(
                        permuted_export, permuted_expected, rtol=0, atol=0)
            # Raw ties resolve by item ID, with identical ranks for duplicates.
            tied_candidates = torch.tensor([[7, 2, 7, 5]], device=device)
            zeros = torch.zeros((1, 4), device=device)
            tied_scores = rank_fusion(zeros, zeros, tied_candidates)
            torch.testing.assert_close(
                tied_scores, torch.tensor([[-2 / 3, 0, -2 / 3, -1 / 3]],
                                          device=device))
            assert module.predict(0, [], [], dataset_name=name) == []
        finally:
            model.chunk_size = saved_chunk_size
            module._MODELS.clear()


def main():
    seed_everything(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    check_purchase_dataset()
    check_attentive_similarity()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Validation: supplied holdout; seed=42; up to 200 users/category; "
        "first heldout item versus 99 unique unseen negatives; "
        "HR@10=mean(rank<10); macro=equal category mean; "
        "expert ties resolved by item ID; fixed equal normalized-rank fusion; "
        "shared ensemble.predict for validation and exported inference.",
        flush=True,
    )
    write_predict_script()
    metrics = [train_category(name, device) for name in DATASETS]
    for name, hr in zip(DATASETS, metrics):
        print(f"{name} selected_validation_HR@10={hr:.5f}", flush=True)
    score = float(np.mean(metrics))
    print(f"Equal-weight macro validation HR@10={score:.5f}", flush=True)
    print(f"Measured ensemble versus similarity reference 0.4725: {score - 0.4725:+.5f}; "
          f"versus SASRec reference 0.5050: {score - 0.5050:+.5f}", flush=True)
    print(f"Final Validation Score: {score}", flush=True)


if __name__ == "__main__":
    import traceback

    os.makedirs(WORKING, exist_ok=True)
    try:
        main()
    except Exception:
        with open(os.path.join(WORKING, "training_error.log"), "w") as f:
            f.write(traceback.format_exc())
        raise
