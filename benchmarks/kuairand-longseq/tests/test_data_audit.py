import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_data.py"
spec = importlib.util.spec_from_file_location("audit_data", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DataAuditTests(unittest.TestCase):
    def run_audit(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sasrec_format.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"])
                writer.writerows(rows)
            return module.audit(path, max_history=2)

    def test_exact_window_and_repeated_target(self):
        report = self.run_audit([[7, "0,1,2,1", "4,4,4,4", "1,2,3,4"]])
        self.assertTrue(report["valid"])
        self.assertEqual(report["train_supervised_positions_per_epoch"], 2)
        self.assertEqual(report["holdout_target_repeated_in_model_history"], 1)
        self.assertFalse(report["paper_statistics_match"])

    def test_repetition_outside_truncated_history(self):
        report = self.run_audit([[7, "0,1,2,0", "4,4,4,4", "1,2,3,4"]])
        self.assertEqual(report["holdout_target_repeated_in_model_history"], 0)

    def test_invalid_temporal_order_is_rejected(self):
        report = self.run_audit([[7, "0,1,2", "4,4,4", "1,3,2"]])
        self.assertFalse(report["valid"])
        self.assertEqual(report["invalid_rows"], 1)

    def test_noncontiguous_ids_are_rejected(self):
        report = self.run_audit([[7, "0,2,2", "4,4,4", "1,2,3"]])
        self.assertFalse(report["raw_item_ids_contiguous_zero_based"])

    def test_duplicate_users_are_rejected(self):
        row = [7, "0,1,2", "4,4,4", "1,2,3"]
        self.assertFalse(self.run_audit([row, row])["valid"])

    def test_nonfinite_rating_is_rejected(self):
        report = self.run_audit([[7, "0,1,2", "4,nan,4", "1,2,3"]])
        self.assertFalse(report["valid"])


if __name__ == "__main__":
    unittest.main()
