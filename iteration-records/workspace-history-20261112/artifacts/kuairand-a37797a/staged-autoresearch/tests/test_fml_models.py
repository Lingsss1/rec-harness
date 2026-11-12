"""Optional real PyTorch integration on tiny generated arrays, not FML scores.

Requires ML dependencies and STAGED_FML_TEST_UPSTREAM / STAGED_FML_TEST_DOMAINBED
pointing at the pinned official checkouts. No download/API/GPU is performed here.
"""
import ast
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(x) is not None for x in ("torch", "torchvision", "numpy"))
UPSTREAM = os.environ.get("STAGED_FML_TEST_UPSTREAM")
DOMAINBED = os.environ.get("STAGED_FML_TEST_DOMAINBED")


@unittest.skipUnless(AVAILABLE and UPSTREAM and DOMAINBED, "optional pinned-upstream real-model integration environment")
class RealFMLModels(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        from staged_research.fml.bootstrap import public_sources
        self.sources = public_sources(self.root, Path(UPSTREAM), Path(DOMAINBED))

    def development(self, task):
        import numpy as np
        import torch
        from staged_research.fml.data import save_arrays
        path = self.root / task / "development"
        path.mkdir(parents=True)
        rng = np.random.RandomState(7)
        names = ("train_env0", "train_env1", "val") if task == "colored_mnist" else ("train", "val")
        for name in names:
            n = 128 if name.startswith("train") else 32
            if task == "colored_mnist":
                x = rng.rand(n, 2, 28, 28).astype(np.float32)
                y = rng.randint(0, 2, n, dtype=np.int64)
            else:
                x = rng.randint(0, 256, (n, 32, 32, 3), dtype=np.uint8)
                y = rng.randint(0, 10, n, dtype=np.int64)
            save_arrays(path, name, (x, y, np.arange(n)))
        if task == "colored_mnist":
            np.save(path / "rng_state.npy", torch.get_rng_state().numpy(), allow_pickle=False)
        if task == "openood":
            save_arrays(path, "ood", (rng.randint(0, 256, (32, 32, 32, 3), dtype=np.uint8), np.zeros(32, dtype=np.int64), np.arange(32)))
        return path

    def roundtrip(self, task):
        from staged_research.fml import evaluator
        from staged_research.fml.smoke import prepare_smoke_data
        from staged_research.storage import inventory, read
        data = self.root / task / "smoke"
        prepare_smoke_data(task, self.development(task), data)
        validation = self.root / task / "validation"
        result = evaluator.evaluate(task, "validation", self.sources[task], data / "validation", validation,
                                    device="cpu", smoke=True)
        self.assertTrue(0 <= result["metric"] <= 1)
        checkpoint = validation / "checkpoint"
        before = inventory(checkpoint)
        original = evaluator.run_worker
        actions = []
        def record(argv, directory, label, isolation):
            actions.append(argv[argv.index("--action") + 1])
            self.assertNotIn("--data", argv)  # no eval label directory passed to predictor
            self.assertFalse(list(directory.glob("*_y.npy")))
            return original(argv, directory, label, isolation)
        with patch.object(evaluator, "run_worker", side_effect=record):
            test = evaluator.evaluate(task, "test", self.sources[task], data / "test", self.root / task / "test",
                                      checkpoint=checkpoint, device="cpu", smoke=True)
        self.assertTrue(0 <= test["metric"] <= 1)
        self.assertTrue(all(action == "infer" for action in actions))
        self.assertEqual(inventory(checkpoint), before)
        return read(checkpoint / "model_spec.json"), result, test

    @unittest.skipUnless(importlib.util.find_spec("timm"), "DomainBed environment needs timm")
    def test_colored_mnist_real_erm_checkpoint_and_heldout_inference(self):
        spec, _, _ = self.roundtrip("colored_mnist")
        self.assertEqual(spec["steps"], 2)
        self.assertEqual(spec["settings"]["workers"], 0)

    def test_openood_real_resnet18_and_external_auroc(self):
        spec, result, _ = self.roundtrip("openood")
        self.assertEqual(spec["training_config"]["epochs"], 1)
        self.assertIn("fpr95", result["details"])

    @unittest.skipUnless(importlib.util.find_spec("opacus"), "Opacus environment needs opacus")
    def test_opacus_real_dp_sgd_checkpoint_and_recomputed_accountant(self):
        from staged_research.fml.evaluator import privacy_checks
        spec, result, _ = self.roundtrip("opacus")
        self.assertTrue(0 < result["constraint_checks"]["actual_epsilon"] <= 8)
        self.assertEqual(spec["logical_steps"], 8)
        # The scorer ignores a forged scalar and checks actual accounted steps.
        spec["epsilon"] = 0.00001
        self.assertEqual(privacy_checks(spec)["actual_epsilon"], result["constraint_checks"]["actual_epsilon"])
        spec["privacy"]["history"][0][2] += 1
        with self.assertRaises(ValueError):
            privacy_checks(spec)

    def test_colored_data_matches_pinned_upstream_transform_and_split(self):
        import numpy as np
        import torch
        from staged_research.fml.data import colored_arrays, seed_hash
        torch.manual_seed(123)
        images = torch.randint(0, 256, (840, 28, 28), dtype=torch.uint8)
        labels = torch.randint(0, 10, (840,))
        class FakeMNIST:
            def __init__(self, root, train, download):
                self.data = images[:600] if train else images[600:]
                self.targets = labels[:600] if train else labels[600:]
        source = Path(UPSTREAM) / "ml_tasks/Generalization_domainbed/original_file_backup/datasets.py"
        parsed = ast.parse(source.read_text())
        classes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in ("MultipleEnvironmentMNIST", "ColoredMNIST")]
        namespace = {"torch": torch, "MNIST": FakeMNIST, "TensorDataset": torch.utils.data.TensorDataset,
                     "MultipleDomainDataset": type("MultipleDomainDataset", (), {})}
        torch.manual_seed(0)
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
        reference = namespace["ColoredMNIST"]("unused", [2], {})
        rng = torch.get_rng_state().numpy().copy()
        actual = colored_arrays(images, labels)
        np.testing.assert_array_equal(actual["rng_state"], rng)
        for env in (0, 1, 2):
            x, y = reference.datasets[env].tensors
            keys = np.arange(len(x))
            np.random.RandomState(seed_hash(0, env)).shuffle(keys)
            boundary = int(len(keys) * 0.3)
            if env < 2:
                keys, name = keys[boundary:], "train_env" + str(env)
            else:
                keys, name = keys[:boundary], "val"
            np.testing.assert_array_equal(actual[name][0], x[keys].numpy())
            np.testing.assert_array_equal(actual[name][1], y[keys].numpy())


if __name__ == "__main__":
    unittest.main()
