"""Fixed training and exact, chunked full-catalog inference for the task adapter."""

from __future__ import annotations

import importlib.util
import json
import random
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from .common import append, digest, file_hash, read, rows, write
from .data import ITEM_FIELDS, USER_ARRAY, USER_SPARSE
from .knowledge import literal_settings


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("taac_candidate_" + file_hash(path), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = module.SETTINGS
    if settings != literal_settings(path):
        raise ValueError(
            "Edit literal SETTINGS directly; runtime settings differ from declarations"
        )
    if settings["negatives"] < 1 or not isinstance(settings["negatives"], int):
        raise ValueError("Positive integer negative count required")
    if settings["temperature"] <= 0:
        raise ValueError("Positive temperature required")
    return module


class FeatureBank:
    def __init__(self, public):
        import numpy as np

        self.public = Path(public)
        self.meta = read(self.public / "metadata.json")
        self.sparse = np.load(self.public / "item_sparse.npy", mmap_mode="r")
        self.mm = np.load(self.public / "item_mm.npy", mmap_mode="r")
        self.mm_seen = np.load(self.public / "mm_seen.npy", mmap_mode="r")
        self.item_seen = np.load(self.public / "item_seen.npy", mmap_mode="r")
        self.c_sparse = np.load(self.public / "candidate_sparse.npy", mmap_mode="r")
        self.c_mm = np.load(self.public / "candidate_mm.npy", mmap_mode="r")
        self.c_rids = np.load(self.public / "candidate_rids.npy", mmap_mode="r")
        self.c_ids = read(self.public / "candidate_ids.json")
        self.users = sqlite3.connect(f"file:{self.public / 'users.sqlite'}?mode=ro", uri=True)
        self.item = lru_cache(maxsize=16384)(self._item)
        self.user = lru_cache(maxsize=4096)(self._user)

    def default(self):
        return {**dict.fromkeys(ITEM_FIELDS + USER_SPARSE, 0), **{k: [] for k in USER_ARRAY}}

    def _item(self, rid):
        return {
            **self.default(),
            **dict(zip(ITEM_FIELDS, map(int, self.sparse[rid]), strict=True)),
            self.meta["mm_emb_id"]: self.mm[rid],
        }

    def _user(self, uid):
        row = self.users.execute("SELECT features FROM users WHERE uid=?", (uid,)).fetchone()
        return {**self.default(), **(json.loads(row[0]) if row else {})}

    def close(self):
        self.item.cache_clear()
        self.user.cache_clear()
        self.users.close()

    def candidate(self, index):
        return {
            **self.default(),
            **dict(zip(ITEM_FIELDS, map(int, self.c_sparse[index]), strict=True)),
            self.meta["mm_emb_id"]: self.c_mm[index],
        }

    def encode_inputs(self, examples, device, training=False):
        import torch

        length = self.meta["history_limit"] + 1
        ids, masks, features, labels, valid = [], [], [], [], []
        for example in examples:
            uid = int(example["user_id"])
            history = example["history"]
            events = history[:-1] if training else history
            events = events[-self.meta["history_limit"] :]
            tokens = [uid] + [e["item_id"] for e in events]
            types = [2] + [1] * len(events)
            # These are observed HISTORY attributes only, never target action/time.
            # The unchanged official model ignores them; future candidate encoders can use them.
            feat = [self.user(uid)] + [
                {
                    **self.item(e["item_id"]),
                    "history_action": e["action_type"],
                    "history_timestamp": e["timestamp"],
                }
                for e in events
            ]
            padding = length - len(tokens)
            ids.append([0] * padding + tokens)
            masks.append([0] * padding + types)
            features.append([self.default() for _ in range(padding)] + feat)
            if training:
                targets = [e["item_id"] for e in history[-len(tokens) :]]
                labels.append([0] * padding + targets)
                valid.append([False] * padding + [True] * len(tokens))
        result = (
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(masks, dtype=torch.long, device=device),
            features,
        )
        if training:
            result += (
                torch.tensor(labels, dtype=torch.long, device=device),
                torch.tensor(valid, dtype=torch.bool, device=device),
            )
        return result


def construct(candidate, bank, device):
    meta = bank.meta
    args = SimpleNamespace(
        **{
            **candidate.SETTINGS,
            "maxlen": meta["history_limit"] + 1,
            "device": device,
            "mm_emb_id": [meta["mm_emb_id"]],
        }
    )
    types = {
        "item_sparse": ITEM_FIELDS,
        "item_array": [],
        "item_continual": [],
        "user_sparse": USER_SPARSE,
        "user_array": USER_ARRAY,
        "user_continual": [],
        "item_emb": [meta["mm_emb_id"]],
    }
    return candidate.BaselineModel(
        meta["n_users"], meta["n_items"], meta["feature_statistics"], types, args
    ).to(device)


def item_vectors(model, ids, features, device):
    import torch

    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    return model.feat2emb(tensor, [features], include_user=False).squeeze(0)


def train(public, source, output, protocol, device, seed):
    import numpy as np
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no silent device fallback")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(protocol.get("cpu_threads", 1))
    bank = FeatureBank(public)
    candidate = load_candidate(source)
    model = construct(candidate, bank, device)
    for _, param in model.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.xavier_normal_(param)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
    optimizer = candidate.make_optimizer(model)
    db = sqlite3.connect(f"file:{Path(public) / 'train.sqlite'}?mode=ro", uri=True)
    order = [row[0] for row in db.execute("SELECT id FROM examples ORDER BY id")]
    rng = random.Random(seed)
    negatives = candidate.SETTINGS["negatives"]
    pool = np.flatnonzero(bank.item_seen)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    steps = 0
    batch_size = protocol["batch_size"]
    model.train()
    for epoch in range(protocol["epochs"]):
        rng.shuffle(order)
        for offset in range(0, len(order), batch_size):
            examples = []
            for row_id in order[offset : offset + batch_size]:
                uid, events = db.execute(
                    "SELECT uid,events FROM examples WHERE id=?", (row_id,)
                ).fetchone()
                examples.append({"user_id": str(uid), "history": json.loads(events)})
            seq, mask, feat, targets, valid = bank.encode_inputs(examples, device, training=True)
            representation = model.log2feats(seq, mask, feat)[valid]
            positive_ids = targets[valid].tolist()
            positive = item_vectors(
                model, positive_ids, [bank.item(i) for i in positive_ids], device
            )
            neg_ids = []
            for example, row_valid in zip(examples, valid, strict=True):
                excluded = {e["item_id"] for e in example["history"]}
                if len(pool) <= len(excluded) and all(int(i) in excluded for i in pool):
                    raise ValueError("No eligible negative items for training user")
                for _ in range(int(row_valid.sum().item()) * negatives):
                    item = int(rng.choice(pool))
                    while item in excluded:
                        item = int(rng.choice(pool))
                    neg_ids.append(item)
            neg = item_vectors(model, neg_ids, [bank.item(i) for i in neg_ids], device)
            neg = neg.reshape(len(positive_ids), negatives, -1)
            positive_scores = (representation * positive).sum(-1)
            negative_scores = (representation.unsqueeze(1) * neg).sum(-1)
            loss = candidate.compute_loss(positive_scores, negative_scores)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("Training loss must be a finite scalar")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            append(
                output / "training.jsonl",
                {
                    "step": steps,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
            if protocol.get("max_steps") and steps >= protocol["max_steps"]:
                break
        if protocol.get("max_steps") and steps >= protocol["max_steps"]:
            break
    if not steps:
        raise ValueError("No training updates executed")
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    seconds = time.monotonic() - start
    torch.save(
        {
            "state_dict": model.state_dict(),
            "settings": candidate.SETTINGS,
            "source_sha256": file_hash(source),
            "metadata_hash": digest(bank.meta),
            "protocol": protocol,
            "seed": seed,
        },
        output / "checkpoint.pt",
    )
    receipt = {
        "steps": steps,
        "training_seconds": seconds,
        "seed": seed,
        "device": device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "source_sha256": file_hash(source),
        "settings": candidate.SETTINGS,
        "protocol": protocol,
        "checkpoint_sha256": file_hash(output / "checkpoint.pt"),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
        if device.startswith("cuda")
        else None,
    }
    write(output / "training.json", receipt)
    db.close()
    bank.close()
    return receipt


def exact_topk(query, candidates, k=10, chunk_size=8192):
    """Exact inner-product retrieval; deterministic candidate-order tie breaking."""
    import torch

    if candidates.shape[0] < k or chunk_size < 1:
        raise ValueError("Invalid full-catalog retrieval dimensions")
    values = torch.empty((len(query), 0), device=query.device)
    indices = torch.empty((len(query), 0), device=query.device, dtype=torch.long)
    for start in range(0, len(candidates), chunk_size):
        scores = query @ candidates[start : start + chunk_size].to(query.device).T
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite retrieval scores")
        ids = torch.arange(start, start + scores.shape[1], device=query.device)
        ids = ids.unsqueeze(0).expand(len(query), -1)
        values = torch.cat([values, scores], 1)
        indices = torch.cat([indices, ids], 1)
        # Incoming IDs are ordered; retained earlier chunks precede later IDs on ties.
        ordering = torch.argsort(values, dim=1, descending=True, stable=True)[:, :k]
        values = values.gather(1, ordering)
        indices = indices.gather(1, ordering)
    return indices, values


def predict(public, source, checkpoint, queries, output, device, batch_size=64, chunk_size=8192):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    bank = FeatureBank(public)
    candidate = load_candidate(source)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["source_sha256"] != file_hash(source) or saved["metadata_hash"] != digest(bank.meta):
        raise ValueError("Checkpoint source/data binding mismatch")
    if saved["settings"] != candidate.SETTINGS:
        raise ValueError("Candidate settings differ from training")
    model = construct(candidate, bank, device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    vectors = []
    with torch.no_grad():
        for offset in range(0, len(bank.c_ids), chunk_size):
            stop = min(offset + chunk_size, len(bank.c_ids))
            v = item_vectors(
                model,
                bank.c_rids[offset:stop].tolist(),
                [bank.candidate(i) for i in range(offset, stop)],
                device,
            )
            vectors.append(v.cpu())
        candidates = torch.cat(vectors)
        np.save(output / "candidate_vectors.npy", candidates.numpy())
        batch = []

        def emit(examples):
            seq, mask, feat = bank.encode_inputs(examples, device)
            q = model.predict(seq, feat, mask)
            top, scores = exact_topk(q, candidates, chunk_size=chunk_size)
            for i, example in enumerate(examples):
                append(
                    output / "predictions.jsonl",
                    {
                        "user_id": example["user_id"],
                        "items": [bank.c_ids[j] for j in top[i].tolist()],
                    },
                )
                append(
                    output / "query_vectors.jsonl",
                    {
                        "user_id": example["user_id"],
                        "vector": q[i].cpu().tolist(),
                        "top_scores": scores[i].cpu().tolist(),
                    },
                )

        for example in rows(queries):
            batch.append(example)
            if len(batch) >= batch_size:
                emit(batch)
                batch = []
        if batch:
            emit(batch)
    write(
        output / "inference.json",
        {
            "seconds": time.monotonic() - started,
            "retrieval": "exact_inner_product_full_candidate_pool",
            "candidate_count": len(bank.c_ids),
            "checkpoint_sha256": file_hash(checkpoint),
            "source_sha256": file_hash(source),
        },
    )
    bank.close()
    return str(output / "predictions.jsonl")
