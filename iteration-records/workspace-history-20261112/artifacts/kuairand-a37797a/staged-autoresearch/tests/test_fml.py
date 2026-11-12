"""Built-in evaluator contracts. Lightweight tests do not train or download."""
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from staged_research.fml import bootstrap
from staged_research.fml.data import manifest, seed_hash, verify_data
from staged_research.fml.specs import COUNTS, DATA_PROTOCOL
from staged_research.storage import ProtocolError, digest, inventory, read, write

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class BootstrapTests(unittest.TestCase):
    def test_plan_has_no_install_network_or_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp) / "absent"
            with patch.object(bootstrap, "command", side_effect=AssertionError("no commands")), \
                 patch.object(bootstrap, "download", side_effect=AssertionError("no download")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bootstrap.main(["--prefix", str(prefix), "--stage", "plan"]), 0)
            self.assertFalse(prefix.exists())

    def test_installer_never_inherits_model_credentials(self):
        import os
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "SYNTHETIC", "OPENAI_API_KEY": "SYNTHETIC"}), \
             patch.object(bootstrap, "invoke_process", return_value={"returncode": 0, "timed_out": False, "stdout": "ok", "stderr": ""}) as run:
            bootstrap.command(["true"])
        env = run.call_args.kwargs["env"]
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)

    def test_smoke_preparation_uses_task_environment_not_controller_numpy(self):
        from staged_research.fml import smoke
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp)
            config = {"agent": {}, "execution": {"gpu_device": "0", "environment_id": "fixture"},
                "protocol": {"gpu_device": "0"}, "tasks": {"colored_mnist": {
                    "public_source": str(prefix / "source"), "validation_data": str(prefix / "development"),
                    "validation_argv": ["/task-env/bin/python"], "test_argv": ["/task-env/bin/python"]}}}
            def evaluate(card, candidate, output, phase, checkpoint):
                write(output / "metrics.json", {"metric": 0.5})
                return {"returncode": 0, "timed_out": False}
            with patch.object(bootstrap, "command", return_value="") as command, \
                 patch.object(smoke, "prepare_smoke_data", side_effect=AssertionError("controller has no numpy")), \
                 patch.object(smoke, "LocalRuntime") as runtime, \
                 patch.object(smoke, "score_result", return_value=0.5):
                runtime.return_value.evaluate.side_effect = evaluate
                receipt = smoke.run_smoke(config, prefix)
            self.assertEqual(command.call_args.args[0][:3], ["/task-env/bin/python", "-m", "staged_research.fml.smoke"])
            self.assertTrue(receipt["passed"])
            self.assertFalse(receipt["hidden_test_used"])
            self.assertEqual(config["tasks"]["colored_mnist"]["test_argv"], ["/task-env/bin/python"])

    def test_calibration_unknown_outcome_is_not_retrained(self):
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp)
            write(prefix / "calibration/colored_mnist/validation.reserved.json", {"job": "reserved"})
            config = {"agent": {}, "execution": {"gpu_device": "0"},
                "tasks": {"colored_mnist": {}}, "protocol": {"gpu_device": "0"}}
            with patch.object(bootstrap, "LocalRuntime") as runtime:
                with self.assertRaisesRegex(ProtocolError, "unknown"):
                    bootstrap.calibrate(prefix, config)
                runtime.return_value.stop.assert_called_once_with("reserved")
                runtime.return_value.evaluate.assert_not_called()

    def test_upstream_changed_commit_is_rejected_without_checkout_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(bootstrap, "command", side_effect=["wrong", ""]):
                with self.assertRaisesRegex(ProtocolError, "pinned commit"):
                    bootstrap.checkout("unused", "f" * 40, directory)

    def test_full_manifest_requires_all_partitions(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "fixture").write_text("not ML data")
            manifest(directory, "opacus", "validation", "full", {"train": 34997})
            with self.assertRaisesRegex(ProtocolError, "partitions"):
                verify_data(directory, "opacus", "validation")

    def test_manifest_detects_file_tampering_and_smoke_cannot_register(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "fixture").write_text("SYNTHETIC")
            manifest(directory, "opacus", "validation", "smoke", {"train": 2, "val": 1})
            with self.assertRaisesRegex(ProtocolError, "Smoke"):
                verify_data(directory, "opacus", "validation")
            self.assertEqual(verify_data(directory, "opacus", "validation", True)["profile"], "smoke")
            (directory / "fixture").write_text("CHANGED")
            with self.assertRaisesRegex(ProtocolError, "contents changed"):
                verify_data(directory, "opacus", "validation", True)


@unittest.skipUnless(HAS_NUMPY, "numpy optional for task adapter unit tests")
class SplitAndScoringTests(unittest.TestCase):
    def test_opacus_exact_official_split_counts_and_disjoint_ids(self):
        import numpy as np
        from staged_research.fml.data import carve_cifar
        x = np.arange(50000).reshape(-1, 1)
        splits = carve_cifar("opacus", x, np.arange(50000) % 10)
        self.assertEqual({k: len(v[0]) for k, v in splits.items()}, COUNTS["opacus"])
        sets = [set(v[2]) for v in splits.values()]
        self.assertEqual(len(set.union(*sets)), 50000)
        for i in range(3):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])
        reference_hidden = np.random.RandomState(42).permutation(50000)[:10500]
        np.testing.assert_array_equal(splits["test"][2], reference_hidden)

    def test_openood_exact_hidden_carve_and_fixed_visible_pool(self):
        import numpy as np
        from staged_research.fml.data import carve_cifar
        splits = carve_cifar("openood", np.zeros((50000, 1)), np.zeros(50000), np.ones((10000, 1)), np.zeros(10000))
        self.assertEqual({k: len(v[0]) for k, v in splits.items()}, {"train": 49979, "val": 3021, "test": 7000})
        np.testing.assert_array_equal(splits["test"][2], 50000 + np.random.RandomState(42).permutation(10000)[3000:])
        self.assertFalse(set(splits["train"][2]) & set(splits["test"][2]))
        self.assertFalse(set(splits["val"][2]) & set(splits["test"][2]))

    def test_binary_logit_accuracy_and_invalid_predictions(self):
        from staged_research.fml.evaluator import accuracy
        self.assertEqual(accuracy([[1], [-1], [0]], [1, 0, 0]), 1)
        with self.assertRaises(ProtocolError):
            accuracy([[float("nan")]], [0])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn optional")
    def test_external_auroc_direction_and_nonfinite_rejection(self):
        from staged_research.fml.evaluator import ood_metrics
        result = ood_metrics([0.9, 0.8, 0.7], [0.1, 0.2, 0.3])
        self.assertEqual(result["auroc"], 1)
        with self.assertRaises(ProtocolError):
            ood_metrics([float("inf")], [0])

    def test_hidden_test_never_invokes_training_and_checks_selected_source(self):
        import numpy as np
        from staged_research.fml.evaluator import evaluate, source_identity
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, data, checkpoint = root / "source", root / "test", root / "checkpoint"
            for path in (source, data, checkpoint):
                path.mkdir()
            (source / "algorithm.py").write_text("raise AssertionError('scorer must not import candidate')")
            np.save(data / "test_x.npy", np.zeros((2, 2, 28, 28), dtype=np.float32))
            np.save(data / "test_y.npy", np.array([0, 1]))
            np.save(data / "test_ids.npy", np.array([10, 20]))
            manifest(data, "colored_mnist", "test", "smoke", {"test": 2})
            write(checkpoint / "model_spec.json", {"task": "colored_mnist", "source_sha256": source_identity(source),
                "protocol": DATA_PROTOCOL, "profile": "smoke", "training_seconds": 2})
            (checkpoint / "model.pt").write_text("SYNTHETIC CHECKPOINT")
            actions = []
            def worker(argv, directory, label, isolation):
                self.assertNotIn("train", argv)
                self.assertFalse(list(directory.glob("*_y.npy")))
                np.save(argv[argv.index("--output") + 1], np.array([[2., 0.], [0., 2.]]))
                actions.append(label)
            with patch("staged_research.fml.evaluator.run_worker", side_effect=worker):
                result = evaluate("colored_mnist", "test", source, data, root / "result", checkpoint, "cpu", True)
            self.assertEqual(result["metric"], 1)
            self.assertEqual(actions, ["infer-0"])
            self.assertFalse((root / "result/checkpoint").exists())
            (source / "algorithm.py").write_text("changed source")
            with self.assertRaisesRegex(ProtocolError, "does not belong"):
                evaluate("colored_mnist", "test", source, data, root / "other-result", checkpoint, "cpu", True)


if __name__ == "__main__":
    unittest.main()
