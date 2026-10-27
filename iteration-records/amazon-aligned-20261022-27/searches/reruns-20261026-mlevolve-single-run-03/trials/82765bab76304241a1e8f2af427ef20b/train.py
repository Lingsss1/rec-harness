from __future__ import annotations

import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.checkpoint import checkpoint

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "working"
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
DIM = int(os.environ.get("SASREC_HIDDEN", "64"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
NEGATIVES = 99


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
                u, i = map(int, fields)
                result[u].append(i)
    return dict(result)


def load_data(name):
    train = read_split(Path(DATA_DIR) / f"{name}_train.txt")
    valid = read_split(Path(DATA_DIR) / f"{name}_valid.txt")
    stats_path = Path(DATA_DIR) / "dataset_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as handle:
            stats = json.load(handle)
    if name in stats:
        itemnum = int(stats[name]["itemnum"])
    else:
        itemnum = max(max(items) for items in train.values() if items)
    return train, valid, itemnum


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = self.dropout1(self.conv1(inputs.transpose(1, 2)))
        outputs = self.dropout2(self.conv2(self.relu(outputs)))
        return inputs + outputs.transpose(1, 2)


class SASRec(nn.Module):
    """Causal sequence encoder with tied input/output item embeddings."""

    def __init__(
        self, itemnum, dim=64, maxlen=128, num_blocks=2, num_heads=1, dropout=0.5
    ):
        super().__init__()
        self.maxlen = maxlen
        self.dim = dim
        self.item_emb = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        # Zero initialization preserves the RNG sequence for existing parameters.
        self.item_bias = nn.Embedding(
            itemnum + 1, 1, padding_idx=0,
            _weight=torch.zeros(itemnum + 1, 1),
        )
        self.pos_emb = nn.Embedding(maxlen, dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.attention_layernorms = nn.ModuleList(
            [nn.LayerNorm(dim, eps=1e-8) for _ in range(num_blocks)]
        )
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    dim, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_blocks)
            ]
        )
        self.forward_layernorms = nn.ModuleList(
            [nn.LayerNorm(dim, eps=1e-8) for _ in range(num_blocks)]
        )
        self.forward_layers = nn.ModuleList(
            [PointWiseFeedForward(dim, dropout) for _ in range(num_blocks)]
        )
        self.last_layernorm = nn.LayerNorm(dim, eps=1e-8)
        # Append initialization so all existing parameter initializations are preserved.
        self.memory_projection = nn.Linear(dim, dim)
        self.memory_gate = nn.Linear(2 * dim, dim)

    def log2feats(self, seq):
        positions = torch.arange(seq.shape[1], device=seq.device)
        padding = seq.eq(0).unsqueeze(-1)
        # Pool only the available truncated prefix, before scaling or dropout.
        embeddings = self.item_emb(seq).masked_fill(padding, 0.0)
        counts = seq.ne(0).cumsum(dim=1).unsqueeze(-1).clamp_min(1)
        prefix_mean = embeddings.cumsum(dim=1) / counts
        seqs = embeddings * (self.dim ** 0.5)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions)[None, :, :])
        seqs = seqs.masked_fill(padding, 0.0)
        causal_mask = torch.ones(
            seq.shape[1], seq.shape[1], dtype=torch.bool, device=seq.device
        ).triu(diagonal=1)
        for attention_norm, attention, forward_norm, forward_layer in zip(
            self.attention_layernorms,
            self.attention_layers,
            self.forward_layernorms,
            self.forward_layers,
        ):
            queries = attention_norm(seqs)
            attended, _ = attention(
                queries, seqs, seqs, attn_mask=causal_mask, need_weights=False
            )
            seqs = queries + attended
            seqs = forward_layer(forward_norm(seqs))
            seqs = seqs.masked_fill(padding, 0.0)
        # One jointly trained encoder supplies both dense loss and prediction features.
        memory = torch.tanh(self.memory_projection(prefix_mean))
        memory = memory.masked_fill(padding, 0.0)
        gate = torch.sigmoid(self.memory_gate(torch.cat([seqs, memory], dim=-1)))
        seqs = (1.0 - gate) * seqs + gate * memory
        return self.last_layernorm(seqs).masked_fill(padding, 0.0)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :]
        return (
            (self.item_emb(candidates) * features[:, None, :]).sum(-1)
            + self.item_bias(candidates).squeeze(-1)
        )

    def forward(self, seq, candidates):
        return self.predict(seq, candidates)


def check_memory_encoder():
    # A tiny CPU diagnostic preserves training RNG state and never fits a predictor.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = SASRec(itemnum=8, dim=8, maxlen=6, dropout=0.5).eval()
        with torch.no_grad():
            model.item_emb.weight.copy_(
                torch.arange(72, dtype=torch.float32).reshape(9, 8) / 8
            )
            model.item_emb.weight[0].zero_()
        seq = torch.tensor([
            [0, 0, 1, 2, 3, 4],
            [0, 0, 1, 2, 5, 6],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 7],
        ])
        captured = []
        hook = model.memory_projection.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        try:
            with torch.no_grad():
                features = model.log2feats(seq)
        finally:
            hook.remove()
        # Compare the actual projection input with an explicit masked prefix mean.
        expected = torch.zeros_like(captured[0])
        with torch.no_grad():
            for row in range(seq.shape[0]):
                for end in range(seq.shape[1]):
                    items = seq[row, :end + 1]
                    items = items[items.ne(0)]
                    if items.numel():
                        expected[row, end] = model.item_emb(items).mean(dim=0)
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
        assert torch.isfinite(features).all()
        assert torch.count_nonzero(features[seq.eq(0)]).item() == 0
        # Changing a suffix must not affect any earlier feature or prefix mean.
        torch.testing.assert_close(
            captured[0][0, :4], captured[0][1, :4], rtol=0, atol=0
        )
        torch.testing.assert_close(features[0, :4], features[1, :4])
        with torch.no_grad():
            shortened = model.log2feats(seq[:1, :4])
        torch.testing.assert_close(features[:1, :4], shortened)
        candidates = torch.tensor([[1, 2, 3]]).expand(seq.shape[0], -1)
        scores = model.predict(seq, candidates)
        assert torch.isfinite(scores).all()
        scores.square().mean().backward()
        for layer in (model.memory_projection, model.memory_gate):
            for parameter in layer.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()


def check_exported_predictor(name, model, sequences, candidates, expected, device):
    # Reload the same selected checkpoint through the public exported interface.
    # This checks reconstruction, strict loading, padding, truncation and scoring;
    # no independently fitted model or combined scores are introduced.
    import importlib.util

    devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        spec = importlib.util.spec_from_file_location(
            "sasrec_export_check", WORK / "predict.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            for row in range(len(sequences)):
                history = sequences[row][sequences[row].ne(0)].tolist()
                candidate_list = candidates[row].tolist()
                actual = module.predict(
                    0, history, candidate_list, dataset_name=name
                )
                torch.testing.assert_close(
                    torch.tensor(actual, dtype=torch.float32), expected[row].cpu(),
                    rtol=1e-4, atol=1e-5,
                )
            # Extra leading history must be discarded exactly as in validation.
            history = sequences[0].tolist()
            actual = module.predict(
                0, [1, 2, 3] + history, candidates[0].tolist(), dataset_name=name
            )
            torch.testing.assert_close(
                torch.tensor(actual, dtype=torch.float32), expected[0].cpu(),
                rtol=1e-4, atol=1e-5,
            )
            assert module.predict(0, [], [], dataset_name=name) == []
        finally:
            module._models.clear()


class PrefixDataset(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.seen = [set(h.tolist()) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No training histories with next-item supervision")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        # Include final training transitions frequently, alongside earlier prefixes.
        end = (
            len(history) - 1
            if np.random.random() < 0.5
            else np.random.randint(1, len(history))
        )
        start = max(0, end - self.maxlen)
        prefix = history[start:end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-len(prefix) :] = prefix
        negatives = np.random.randint(
            1, self.itemnum + 1, size=(len(prefix), NEGATIVES)
        )
        seen = self.seen[index]
        if len(seen) >= self.itemnum:
            raise ValueError("Training user has no available negative items")
        for j in range(len(prefix)):
            for k in range(NEGATIVES):
                while int(negatives[j, k]) in seen:
                    negatives[j, k] = np.random.randint(1, self.itemnum + 1)
        candidates = np.zeros((self.maxlen, NEGATIVES + 1), dtype=np.int64)
        candidates[-len(prefix) :, 0] = history[start + 1 : end + 1]
        candidates[-len(prefix) :, 1:] = negatives
        return seq, candidates


def dense_sampled_loss(model, features, candidates, mask):
    # Weight nonpadding transitions uniformly within each prefix.
    # Each nonempty prefix contributes total weight 1 / batch size.
    # Padding rows never enter the embedding lookup or cross-entropy calculation.
    raw_weights = mask.float()
    counts = raw_weights.sum(dim=1).clamp_min(1.0)
    rows = mask.nonzero(as_tuple=True)[0]
    weights = counts[rows].reciprocal() / mask.shape[0]
    features = features[mask]
    candidates = candidates[mask]

    def score_chunk(
        features, candidates, weights, embedding_weight, item_bias_weight
    ):
        # Recompute lookup and scoring on backward instead of retaining every
        # position's 33 output embeddings. All logits and losses use float32.
        with torch.autocast(device_type=features.device.type, enabled=False):
            embeddings = torch.nn.functional.embedding(
                candidates, embedding_weight, padding_idx=0
            ).float()
            logits = torch.bmm(
                embeddings, features.float().unsqueeze(-1)
            ).squeeze(-1)
            logits = logits + torch.nn.functional.embedding(
                candidates, item_bias_weight, padding_idx=0
            ).squeeze(-1).float()
            losses = torch.nn.functional.cross_entropy(
                logits,
                torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
                reduction="none",
                label_smoothing=0.0,
            )
            return (losses * weights).sum()

    loss = features.new_zeros((), dtype=torch.float32)
    for start in range(0, features.shape[0], 512):
        loss = loss + checkpoint(
            score_chunk,
            features[start : start + 512],
            candidates[start : start + 512],
            weights[start : start + 512],
            model.item_emb.weight,
            model.item_bias.weight,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return loss


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def validation_tensors(train, valid, itemnum, maxlen):
    # Preserve input order and Python sampling semantics of the supplied baseline.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        target = valid[user][0]
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Cannot form the required 99 unique validation negatives")
        negative_pool = set()
        while len(negative_pool) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negative_pool.add(item)
        hist = train[user][-maxlen:]
        seq = [0] * maxlen
        if hist:
            seq[-len(hist) :] = hist
        sequences.append(seq)
        candidates.append([target] + list(negative_pool))
    if not sequences:
        raise ValueError("No validation users")
    return torch.tensor(sequences), torch.tensor(candidates)


@torch.inference_mode()
def evaluate(model, tensors, device):
    model.eval()
    hits = 0
    seqs, candidates = tensors
    for start in range(0, len(seqs), 64):
        scores = model.predict(
            seqs[start : start + 64].to(device),
            candidates[start : start + 64].to(device),
        )
        # Stable descending rank: target occupies candidate index zero.
        ranks = (scores[:, 1:] > scores[:, :1]).sum(dim=1)
        hits += int((ranks < 10).sum().item())
    return hits / len(seqs)


def train_category(name, device, seconds_available):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    dataset = PrefixDataset(train, itemnum, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=BATCH * STEPS, generator=generator
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
    )
    config = dict(
        itemnum=itemnum, dim=DIM, maxlen=MAXLEN,
        num_blocks=2, num_heads=1, dropout=0.5,
    )
    model = SASRec(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, betas=(0.9, 0.98), weight_decay=1e-4
    )
    val_tensors = validation_tensors(train, valid, itemnum, MAXLEN)
    # Include real histories, an empty history and a one-item padded history.
    probe_seqs = torch.cat([
        val_tensors[0][:2], torch.zeros((2, MAXLEN), dtype=torch.long)
    ])
    probe_seqs[-1, -1] = val_tensors[1][0, 0]
    probe_candidates = torch.cat([
        val_tensors[1][:2], val_tensors[1][:1].repeat(2, 1)
    ])
    best_probe_scores = None
    checkpoint = WORK / f"{name}_model.pt"
    best, best_epoch = -1.0, 0
    started = time.monotonic()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq, candidates in loader:
            seq = seq.to(device, non_blocking=True)
            candidates = candidates.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                mask = seq.ne(0)
                features = model.log2feats(seq)
                loss = dense_sampled_loss(model, features, candidates, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        elapsed = time.monotonic() - started
        must_stop = elapsed >= seconds_available
        if epoch % 5 == 0 or epoch == EPOCHS or must_stop:
            hr = evaluate(model, val_tensors, device)
            if hr > best:
                best, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
                with torch.inference_mode():
                    best_probe_scores = model.predict(
                        probe_seqs.to(device), probe_candidates.to(device)
                    ).float().cpu()
            print(
                f"{name} epoch={epoch} loss={total_loss / STEPS:.5f} "
                f"validation_HR@10={hr:.5f}",
                flush=True,
            )
            if epoch - best_epoch >= 40 or must_stop:
                break
        else:
            print(f"{name} epoch={epoch} loss={total_loss / STEPS:.5f}", flush=True)
    with open(WORK / f"{name}_meta.pkl", "wb") as handle:
        pickle.dump(config, handle)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    score = evaluate(model, val_tensors, device)
    # Compare the selected model's saved-time scores with its strict reload.
    with torch.inference_mode():
        reloaded_scores = model.predict(
            probe_seqs.to(device), probe_candidates.to(device)
        ).float().cpu()
    torch.testing.assert_close(
        reloaded_scores, best_probe_scores, rtol=1e-5, atol=1e-6
    )
    check_exported_predictor(
        name, model, probe_seqs, probe_candidates, reloaded_scores, device
    )
    print(f"{name} selected_checkpoint_and_predictor_parity=passed", flush=True)
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import pickle
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
            'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = None
_model_class = None


def _load(dataset_name):
    global _device, _model_class
    if dataset_name in _models:
        return _models[dataset_name]
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _model_class is None:
        spec = importlib.util.spec_from_file_location(
            'sasrec_training', str(HERE.parent / 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.SASRec
    with open(HERE / f'{dataset_name}_meta.pkl', 'rb') as handle:
        config = pickle.load(handle)
    # Reconstruct the gated-memory SASRec from the unchanged saved config.
    # Strict loading includes item_bias, memory_projection and memory_gate
    # alongside all existing parameters in this route's single selected checkpoint.
    model = _model_class(**config).to(_device)
    state = torch.load(HERE / f'{dataset_name}_model.pt',
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    return model


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = DATASETS[0]
    model = _load(dataset_name)
    if not candidates:
        return []
    # Match validation: truncate history, right-align, and use model.predict.
    hist = list(history[-model.maxlen:])
    seq = [0] * model.maxlen
    if hist:
        seq[-len(hist):] = hist
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_device)
    return model.predict(seq_t, cand_t)[0].float().cpu().tolist()
"""
    (WORK / "predict.py").write_text(script)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    check_memory_encoder()
    print("Causal gated-memory encoder checks=passed", flush=True)
    write_predict_script()
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    scores = []
    # Reserve headroom below the nine-hour host limit and retain all categories.
    for index, name in enumerate(DATASETS):
        remaining = max(60.0, 8 * 3600 - (time.monotonic() - started))
        allocation = remaining / (len(DATASETS) - index)
        scores.append(train_category(name, device, allocation))
        print(f"{name} selected_validation_HR@10={scores[-1]:.5f}", flush=True)
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")
    print(
        f"Macro validation HR@10={score:.5f}; "
        f"delta_vs_previous_0.4463={score - 0.4463:+.5f}; "
        f"delta_vs_baseline_0.505={score - 0.505:+.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
