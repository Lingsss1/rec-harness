"""Reference-compatible per-user split and reusable memory-mapped cache."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
PAPER_STATS = {"users": 27284, "observed_items": 131090, "interactions": 97010279}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_cache(source, output, max_history=1024, preprocessing_manifest=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Cache directory already exists: " + str(output))
    if max_history < 1:
        raise ValueError("max_history must be positive")
    records, users, items, interactions = [], set(), set(), 0
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            uid = int(row["user_id"])
            for key in ("sequence_item_ids", "sequence_timestamps"):
                if not re.fullmatch(r"\s*-?\d+\s*(,\s*-?\d+\s*)*", row[key]):
                    raise ValueError("Malformed integer sequence for user " + str(uid))
            ids = np.fromstring(row["sequence_item_ids"], sep=",", dtype=np.int64)
            times = np.fromstring(row["sequence_timestamps"], sep=",", dtype=np.int64)
            # Roundtrip token counts catch fromstring's permissive partial parsing.
            if len(ids) != len(row["sequence_item_ids"].split(",")) or len(times) != len(row["sequence_timestamps"].split(",")):
                raise ValueError("Malformed sequence for user " + str(uid))
            if uid in users or len(ids) < 3 or len(ids) != len(times) or np.any(ids < 0):
                raise ValueError("Invalid/duplicate user sequence: " + str(uid))
            if np.any(times[1:] < times[:-1]):
                raise ValueError("Nonchronological timestamps for user " + str(uid))
            users.add(uid)
            items.update(ids.tolist())
            interactions += len(ids)
            records.append((uid, ids[-max_history - 2:] + 1, times[-max_history - 2:]))
    if not records or min(items) != 0 or max(items) + 1 != len(items):
        raise ValueError("Processed item IDs must be zero-based and contiguous")
    output.mkdir(parents=True)
    shape = (len(records), max_history + 2)
    ids_array = np.lib.format.open_memmap(output / "ids.npy", mode="w+", dtype="int64", shape=shape)
    time_array = np.lib.format.open_memmap(output / "times.npy", mode="w+", dtype="int64", shape=shape)
    lengths = np.zeros(len(records), dtype=np.int64)
    uids = np.zeros(len(records), dtype=np.int64)
    for i, (uid, ids, times) in enumerate(records):
        ids_array[i] = 0
        time_array[i] = 0
        ids_array[i, :len(ids)], time_array[i, :len(ids)] = ids, times
        lengths[i], uids[i] = len(ids), uid
    ids_array.flush()
    time_array.flush()
    np.save(output / "lengths.npy", lengths)
    np.save(output / "users.npy", uids)
    stats = dict(users=len(users), observed_items=len(items), interactions=interactions)
    metadata = {
        "format_version": 1, "source_csv_sha256": sha256(source),
        "statistics": stats, "paper_statistics_match": stats == PAPER_STATS,
        "max_history": max_history, "model_item_shift": 1,
        "preprocessing_manifest": preprocessing_manifest,
        "files": {p.name: sha256(p) for p in output.glob("*.npy")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class AlignedDataset(Dataset):
    def __init__(self, cache, split):
        if split not in {"train", "eval"}:
            raise ValueError("Unknown split")
        self.root, self.split = Path(cache), split
        self.meta = json.loads((self.root / "metadata.json").read_text())
        self.max_history = self.meta["max_history"]
        self.ids = np.load(self.root / "ids.npy", mmap_mode="r")
        self.times = np.load(self.root / "times.npy", mmap_mode="r")
        self.lengths = np.load(self.root / "lengths.npy", mmap_mode="r")

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        n = int(self.lengths[index])
        all_ids = self.ids[index, :n]
        all_times = self.times[index, :n]
        if self.split == "train":
            sequence, timestamps = all_ids[:-1], all_times[:-1]
            history, labels = sequence[:-1], sequence[1:]
            target = int(sequence[-1])
        else:
            sequence, timestamps = all_ids[-self.max_history - 1:], all_times[-self.max_history - 1:]
            history = sequence[:-1]
            labels = np.zeros(len(history), dtype=np.int64)
            target = int(sequence[-1])
        length = len(history)
        ids = torch.zeros(self.max_history, dtype=torch.long)
        times = torch.zeros(self.max_history + 1, dtype=torch.long)
        targets = torch.zeros(self.max_history, dtype=torch.long)
        ids[:length] = torch.from_numpy(history.copy())
        times[:length + 1] = torch.from_numpy(timestamps.copy())
        targets[:length] = torch.from_numpy(labels.copy())
        return dict(ids=ids, times=times, labels=targets,
                    length=torch.tensor(length), target=torch.tensor(target),
                    user_index=torch.tensor(index))


def verify_cache(cache):
    root = Path(cache)
    meta = json.loads((root / "metadata.json").read_text())
    for name, expected in meta["files"].items():
        if sha256(root / name) != expected:
            raise ValueError("Changed cache file: " + name)
    return meta
