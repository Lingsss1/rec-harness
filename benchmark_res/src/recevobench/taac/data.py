"""Stream native TencentGR Parquet into an explicitly local, audited protocol."""

from __future__ import annotations

import math
import pickle
import sqlite3
from collections import Counter
from contextlib import suppress
from pathlib import Path

from .common import append, digest, file_hash, inventory, read, write

ITEM_FIELDS = [
    "100",
    "101",
    "102",
    "112",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "120",
    "121",
    "122",
]
USER_SPARSE = ["103", "104", "105", "109"]
USER_ARRAY = ["106", "107", "108", "110"]
PARTITIONS = ("train", "feedback", "gate", "final")


def parquet_rows(path):
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="parquet")
    for batch in dataset.scanner(batch_size=4096).to_batches():
        yield from batch.to_pylist()


def partition(user, salt, fractions):
    if len(fractions) != 4 or any(not math.isfinite(f) or f <= 0 for f in fractions):
        raise ValueError("Four positive finite split fractions required")
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    x = int(digest([salt, str(user)]), 16) / 2**256
    cumulative = 0
    for name, fraction in zip(PARTITIONS, fractions, strict=True):
        cumulative += fraction
        if x < cumulative:
            return name
    return "final"


def query_from_sequence(events, candidates, cut_fraction=0.8):
    """First post-cut click, only with uniquely attributable post-cut exposure.

    Repeated exposures since the preceding click of that ad are ambiguous.
    Histories exclude the target exposure and all events at its timestamp.
    This conservative reconstruction is NOT an official hidden answer set.
    """
    if not 0 < cut_fraction < 1:
        raise ValueError("cut_fraction must lie strictly between zero and one")
    if any(e["action_type"] is not None and e["action_type"] not in (0, 1) for e in events):
        raise ValueError("Only preliminary exposure/click actions 0/1 are supported")
    if any(a["timestamp"] > b["timestamp"] for a, b in zip(events, events[1:], strict=False)):
        raise ValueError("Sequence must already be chronological; not silently sorted")
    # The public release contains null action labels. They cannot be assumed to
    # be exposures or clicks: either interpretation can change the first click
    # or the exposure attribution. Retain them in training, but conservatively
    # exclude this user's held-out query and account for the lost coverage.
    if any(e["action_type"] is None for e in events):
        return None, "unknown_action"
    cut = int(len(events) * cut_fraction)
    j = next((i for i in range(cut, len(events)) if events[i]["action_type"] == 1), None)
    if j is None:
        return None, "no_post_cut_click"
    target = events[j]["item_id"]
    if target not in candidates:
        return None, "target_outside_candidates"
    previous_click = max(
        (i for i in range(j) if events[i]["item_id"] == target and events[i]["action_type"] == 1),
        default=-1,
    )
    exposures = [
        i
        for i in range(previous_click + 1, j)
        if events[i]["item_id"] == target and events[i]["action_type"] == 0
    ]
    if len(exposures) != 1:
        return None, "ambiguous_or_missing_exposure"
    exposure = exposures[0]
    if exposure < cut:
        return None, "exposure_before_cut"
    ts = events[exposure]["timestamp"]
    history = [e for e in events[:exposure] if e["timestamp"] < ts]
    if not history:
        return None, "empty_history"
    return {"history": history, "target_rid": target, "exposure_timestamp": ts}, None


def _lookup(mapping, value, default=None):
    result = mapping.get(value, mapping.get(str(value)))
    if result is None:
        with suppress(TypeError, ValueError):
            result = mapping.get(int(value))
    return default if result is None else result


def prepare(config_path, output):
    import numpy as np

    config = read(config_path)
    raw = Path(config["raw_root"]).expanduser().resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Prepared output must not already exist")
    if not config.get("release"):
        raise ValueError("An explicit data release is required")
    # Pickle is executable; bind the exact trusted publisher file before loading.
    index_path = raw / "indexer.pkl"
    if file_hash(index_path) != config.get("indexer_sha256"):
        raise ValueError("Set indexer_sha256 to the independently checked trusted indexer file")
    with index_path.open("rb") as stream:
        indexer = pickle.load(stream)  # trusted, explicitly hash-bound upstream asset
    item_map = indexer["i"]
    user_map = indexer["u"]
    n_items = max(map(int, item_map.values()))
    n_users = max(map(int, user_map.values()))
    if min(map(int, item_map.values())) < 1 or min(map(int, user_map.values())) < 1:
        raise ValueError("RID zero is reserved for padding/unknown")
    stats = {
        k: max(map(int, indexer["f"][k].values()), default=0)
        for k in ITEM_FIELDS + USER_SPARSE + USER_ARRAY
    }
    mm = config.get("mm_emb_id", "81")
    dims = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    dimension = dims[mm]
    source_paths = [raw / x for x in ["seq", "candidate", "item_feat", "user_feat"]]
    mm_path = raw / "mm_emb" / f"emb_{mm}_{dimension}_parquet"
    source_paths += [mm_path]
    raw_files = [p for root in source_paths for p in sorted(root.rglob("*.parquet"))]
    if any(not list(p.rglob("*.parquet")) for p in source_paths):
        raise ValueError("All native Parquet directories, including multimodal features, required")
    raw_identity = {str(p.relative_to(raw)): file_hash(p) for p in raw_files + [index_path]}
    fractions = config.get("fractions", [0.85, 0.05, 0.05, 0.05])
    partition("validate", config["split_salt"], fractions)
    cut_fraction = config.get("cut_fraction", 0.8)
    history_limit = config.get("history_limit", 100)
    if not isinstance(history_limit, int) or not 1 <= history_limit <= 100:
        raise ValueError("Initial protocol history_limit must be 1..100")
    output.mkdir(parents=True)
    public = output / "public"
    private = output / "private"
    public.mkdir()
    private.mkdir(mode=0o700)

    def array(name, shape, dtype):
        return np.lib.format.open_memmap(public / name, mode="w+", dtype=dtype, shape=shape)

    item_sparse = array("item_sparse.npy", (n_items + 1, len(ITEM_FIELDS)), "int64")
    item_seen = array("item_seen.npy", (n_items + 1,), "bool")
    item_mm = array("item_mm.npy", (n_items + 1, dimension), "float32")
    mm_seen = array("mm_seen.npy", (n_items + 1,), "bool")
    counts = Counter()

    def sparse_value(row, field):
        v = row.get(field)
        if isinstance(v, dict):
            # Public candidate feature_value is already an RID when cold_start=0.
            if v.get("cold_start") != 0:
                return 0
            v = v.get("feature_value")
        v = 0 if v is None else int(v)
        if not 0 <= v <= stats[field]:
            raise ValueError("Feature RID outside indexer vocabulary: " + field)
        return v

    for row in parquet_rows(raw / "item_feat"):
        rid = int(row["item_id"])
        if not 1 <= rid <= n_items or item_seen[rid]:
            raise ValueError("Invalid or duplicate item RID")
        item_sparse[rid] = [sparse_value(row, k) for k in ITEM_FIELDS]
        item_seen[rid] = True
    # Candidate OIDs and retrieval IDs remain separate from model RIDs.
    candidate_rows = list(parquet_rows(raw / "candidate"))
    candidate_rows.sort(key=lambda r: int(r["retrieval_id"]))
    n_candidates = len(candidate_rows)
    if n_candidates < 10 or len({r["item_id"] for r in candidate_rows}) != n_candidates:
        raise ValueError("At least ten distinct candidate OIDs required")
    if len({r["retrieval_id"] for r in candidate_rows}) != n_candidates:
        raise ValueError("Duplicate candidate retrieval IDs")
    fixture = config.get("fixture", False)
    if not fixture and n_candidates != 660000:
        raise ValueError("TencentGR-1M requires the full 660000-ad candidate pool")
    candidate_ids = [str(r["item_id"]) for r in candidate_rows]
    candidate_rids = [int(_lookup(item_map, r["item_id"], 0)) for r in candidate_rows]
    candidate_mm = array("candidate_mm.npy", (n_candidates, dimension), "float32")
    candidate_sparse = array("candidate_sparse.npy", (n_candidates, len(ITEM_FIELDS)), "int64")
    for i, row in enumerate(candidate_rows):
        candidate_sparse[i] = [sparse_value(row, k) for k in ITEM_FIELDS]
    candidate_oid_index = {oid: i for i, oid in enumerate(candidate_ids)}
    candidate_seen = np.zeros(n_candidates, dtype=bool)
    for row in parquet_rows(mm_path):
        oid = str(row["anonymous_cid"])
        if row["emb"] is None:
            # A publisher null is an absent feature, not a malformed numeric
            # vector. Keep the zero default and the coverage mask false.
            counts["missing_mm_vectors"] += 1
            counts["candidate_missing_mm_vectors"] += oid in candidate_oid_index
            continue
        vector = np.asarray(row["emb"], dtype=np.float32)
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError("Invalid multimodal embedding")
        rid = _lookup(item_map, oid)
        ci = candidate_oid_index.get(oid)
        if rid is not None:
            if mm_seen[int(rid)]:
                raise ValueError("Duplicate multimodal OID")
            item_mm[int(rid)] = vector
            mm_seen[int(rid)] = True
        if ci is not None:
            if candidate_seen[ci]:
                raise ValueError("Duplicate candidate multimodal OID")
            candidate_mm[ci] = vector
            candidate_seen[ci] = True
    np.save(public / "candidate_rids.npy", np.array(candidate_rids, dtype=np.int64))
    write(public / "candidate_ids.json", candidate_ids)
    users = sqlite3.connect(public / "users.sqlite")
    users.execute("CREATE TABLE users (uid INTEGER PRIMARY KEY, features TEXT NOT NULL)")
    import json

    for row in parquet_rows(raw / "user_feat"):
        uid = int(row["user_id"])
        if not 1 <= uid <= n_users:
            raise ValueError("User RID outside indexer")
        feature = {k: sparse_value(row, k) for k in USER_SPARSE}
        for k in USER_ARRAY:
            vals = row.get(k) or []
            if any(not 0 <= int(v) <= stats[k] for v in vals):
                raise ValueError("User array RID outside vocabulary")
            feature[k] = [int(v) for v in vals]
        users.execute("INSERT INTO users VALUES (?,?)", (uid, json.dumps(feature)))
    users.commit()
    users.close()
    train = sqlite3.connect(public / "train.sqlite")
    train.execute("CREATE TABLE examples (id INTEGER PRIMARY KEY, uid INTEGER UNIQUE, events TEXT)")
    split_db = sqlite3.connect(output / "split-audit.sqlite")
    split_db.execute("CREATE TABLE users (uid INTEGER PRIMARY KEY, partition TEXT)")
    rid_to_oid = {rid: candidate_ids[i] for i, rid in enumerate(candidate_rids) if rid}
    if len(rid_to_oid) != sum(bool(r) for r in candidate_rids):
        raise ValueError("Multiple candidate OIDs map to the same nonzero RID")
    for p in PARTITIONS[1:]:
        (public / f"{p}.jsonl").touch()
        (private / f"{p}.jsonl").touch()
    for row in parquet_rows(raw / "seq"):
        uid = int(row["user_id"])
        if not 1 <= uid <= n_users:
            raise ValueError("Sequence user outside indexer")
        events = [
            {
                "item_id": int(e["item_id"]),
                "action_type": None if e["action_type"] is None else int(e["action_type"]),
                "timestamp": int(e["timestamp"]),
            }
            for e in row["seq"]
        ]
        if any(not 1 <= e["item_id"] <= n_items for e in events):
            raise ValueError("Sequence item outside indexer")
        # Validate even training rows; no silent sorting, non-preliminary actions or truncation.
        if len(events) > 100:
            raise ValueError("TencentGR-1M sequence longer than 100")
        query, reason = query_from_sequence(events, rid_to_oid, cut_fraction)
        missing_actions = sum(e["action_type"] is None for e in events)
        counts["unknown_action_events"] += missing_actions
        counts["users_with_unknown_action"] += bool(missing_actions)
        part = partition(uid, config["split_salt"], fractions)
        split_db.execute("INSERT INTO users VALUES (?,?)", (uid, part))
        counts[part + "_users"] += 1
        if part == "train":
            if len(events) >= 2:
                train.execute(
                    "INSERT INTO examples(uid,events) VALUES (?,?)", (uid, json.dumps(events))
                )
                counts["train_examples"] += 1
        elif query is None:
            counts[part + "_excluded_" + reason] += 1
        else:
            append(
                public / f"{part}.jsonl",
                {"user_id": str(uid), "history": query["history"][-history_limit:]},
            )
            append(
                private / f"{part}.jsonl",
                {"user_id": str(uid), "target": rid_to_oid[query["target_rid"]]},
            )
            counts[part + "_queries"] += 1
    train.commit()
    train.close()
    split_db.commit()
    split_db.close()
    if not counts["train_examples"] or any(not counts[p + "_queries"] for p in PARTITIONS[1:]):
        raise ValueError(
            "Empty train/feedback/gate/final: inspect exclusions; do not change metric"
        )
    for a in (item_sparse, item_seen, item_mm, mm_seen, candidate_mm, candidate_sparse):
        a.flush()
    del item_sparse, item_seen, item_mm, mm_seen, candidate_mm, candidate_sparse
    metadata = {
        "schema": "taac-local-v1",
        "release": config["release"],
        "fixture": fixture,
        "n_items": n_items,
        "n_users": n_users,
        "n_candidates": n_candidates,
        "feature_statistics": stats,
        "mm_emb_id": mm,
        "mm_dimension": dimension,
        "history_limit": history_limit,
        "fractions": fractions,
        "split_salt": config["split_salt"],
        "cut_fraction": cut_fraction,
        "counts": dict(counts),
        "candidate_mm_coverage": float(candidate_seen.mean()),
        "training_supervision": "next_event (official-code adaptation)",
        "action_null_policy": "Preserve null actions in training; exclude held-out users with any null action",
        "evaluation": "local first post-cut click, unique post-cut exposure; not leaderboard",
        "weights": {"HitRate@10": 0.31, "NDCG@10": 0.69},
    }
    write(public / "metadata.json", metadata)
    write(
        output / "source.json",
        {"config": config, "files": raw_identity, "identity": digest(raw_identity)},
    )
    files = inventory(output)
    write(output / "manifest.json", {"files": files, "identity": digest(files)})
    return metadata
