import csv
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

from longseq_bench.data import AlignedDataset, make_cache, verify_cache
from longseq_bench.prepare import LOGS, preprocess_author_code
from longseq_bench.train import rng_state, restore_rng

REFERENCE = Path(__file__).resolve().parents[1] / "vendor/fuxi-linear"


class PipelineTests(unittest.TestCase):
    def write_csv(self, path):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"])
            for user, values in enumerate([list(range(12)), [0, 1, 2]]):
                writer.writerow([user, ",".join(map(str, values)), ",".join(["4"] * len(values)),
                                 ",".join(str(1000 * x) for x in range(len(values)))])

    @unittest.skipUnless(REFERENCE.exists() and importlib.util.find_spec("pandas") and importlib.util.find_spec("tqdm"),
                         "Requires pinned reference and pandas/tqdm")
    def test_split_ids_lengths_targets_and_timestamps_equal_author_dataset(self):
        sys.path.insert(0, str(REFERENCE))
        from generative_recommenders.data.dataset import DatasetV2
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sequences.csv"
            self.write_csv(source)
            make_cache(source, root / "cache", max_history=4)
            for split, ignore in (("train", 1), ("eval", 0)):
                own = AlignedDataset(root / "cache", split)
                reference = DatasetV2(str(source), padding_length=5, ignore_last_n=ignore,
                                      shift_id_by=1, chronological=True)
                for index in range(len(own)):
                    actual, expected = own[index], reference[index]
                    torch.testing.assert_close(actual["ids"], expected["historical_ids"])
                    self.assertEqual(int(actual["length"]), expected["history_lengths"])
                    self.assertEqual(int(actual["target"]), expected["target_ids"])
                    n = int(actual["length"])
                    torch.testing.assert_close(actual["times"][:n], expected["historical_timestamps"][:n])
                    self.assertEqual(int(actual["times"][n]), expected["target_timestamps"])
                    if split == "train":
                        labels = torch.cat((expected["historical_ids"][1:n], torch.tensor([expected["target_ids"]])))
                        torch.testing.assert_close(actual["labels"][:n], labels)

    def test_cache_hash_detects_modified_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_csv(root / "source.csv")
            make_cache(root / "source.csv", root / "cache", max_history=4)
            verify_cache(root / "cache")
            with (root / "cache/ids.npy").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaises(ValueError):
                verify_cache(root / "cache")

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "Requires pandas")
    def test_released_preprocessing_keeps_df1_and_iterative_core(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for user, item, n in ((0, 10, 350), (1, 20, 350), (2, 10, 10), (0, 30, 349)):
                rows.extend([[user, item, len(rows) + i, 0, 1] for i in range(n)])
            for i, name in enumerate(LOGS):
                with (root / name).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["user_id", "video_id", "time_ms", "play_time_ms", "is_hate"])
                    writer.writerows(rows[i::4])
            output = root / "processed.csv"
            manifest = preprocess_author_code(root, output)
            with output.open(newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual([int(row["user_id"]) for row in records], [0, 1])
            self.assertEqual([len(row["sequence_item_ids"].split(",")) for row in records], [350, 350])
            self.assertEqual(records[0]["sequence_item_ids"], ",".join(["0"] * 350))
            self.assertFalse(manifest["positive_feedback_filter_applied"])

    def test_rng_state_restores_training_randomness(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        state = rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(state)
        actual = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])


if __name__ == "__main__":
    unittest.main()
