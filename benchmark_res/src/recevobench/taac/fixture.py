"""Native-format synthetic fixtures for real CPU integration tests, not task scores."""

import pickle
from pathlib import Path

from .common import file_hash, write
from .data import ITEM_FIELDS, USER_ARRAY, USER_SPARSE


def make_fixture(root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = root / "raw"
    raw.mkdir()

    def table(name, records):
        p = raw / name
        p.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(records), p / "part-0.parquet")

    item_map = {20_000_000_000 + i: i for i in range(1, 17)}
    indexer = {
        "i": item_map,
        "u": {10_000 + i: i for i in range(1, 129)},
        "f": {k: {str(i): i for i in range(1, 5)} for k in ITEM_FIELDS + USER_ARRAY + USER_SPARSE},
    }
    with (raw / "indexer.pkl").open("wb") as stream:
        pickle.dump(indexer, stream)
    table(
        "item_feat", [{"item_id": i, **{k: i % 4 + 1 for k in ITEM_FIELDS}} for i in range(1, 17)]
    )
    table(
        "candidate",
        [
            {
                "item_id": oid,
                "retrieval_id": i - 1,
                **{k: {"cold_start": 0, "feature_value": str(i % 4 + 1)} for k in ITEM_FIELDS},
            }
            for oid, i in item_map.items()
        ],
    )
    table(
        "user_feat",
        [
            {
                "user_id": i,
                **{k: i % 4 + 1 for k in USER_SPARSE},
                **{k: [i % 4 + 1] for k in USER_ARRAY},
            }
            for i in range(1, 129)
        ],
    )
    table(
        "mm_emb/emb_81_32_parquet",
        [
            {"anonymous_cid": str(oid), "emb": [((i + j) % 7) / 7 for j in range(32)]}
            for oid, i in item_map.items()
        ],
    )
    sequences = []
    for uid in range(1, 129):
        target = 5 + uid % 12
        items = [1, 1, 2, 2, 3, 3, target, target]
        sequences.append(
            {
                "user_id": uid,
                "seq": [
                    {"item_id": i, "action_type": j % 2, "timestamp": j + 1}
                    for j, i in enumerate(items)
                ],
            }
        )
    table("seq", sequences)
    config = {
        "raw_root": str(raw),
        "release": "synthetic-native-integration-fixture",
        "indexer_sha256": file_hash(raw / "indexer.pkl"),
        "fixture": True,
        "split_salt": "taac-fixture-1",
        "fractions": [0.5, 0.2, 0.15, 0.15],
        "cut_fraction": 0.75,
        "history_limit": 8,
        "mm_emb_id": "81",
    }
    write(root / "prepare.json", config)
    return str(root / "prepare.json")
