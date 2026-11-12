import tempfile
import unittest
from pathlib import Path

from rec_harness.core import ContractError, file_hash, write_json
from rec_harness.cpu_launcher import validate_launch


class CPULaunchTests(unittest.TestCase):
    def test_gpu_exposure_and_wrong_identity_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dev').mkdir()
            entry = root / 'probe.py'; entry.write_text('raise AssertionError("must not run")')
            write_json(root / 'jail-manifest.json', {'uid': 12345, 'gpu_devices': [0]})
            with self.assertRaisesRegex(ContractError, 'no GPU'):
                validate_launch(root, '/probe.py', file_hash(entry), 12345)
            write_json(root / 'jail-manifest.json', {'uid': 12345, 'gpu_devices': []})
            (root / 'dev/nvidia0').touch()
            with self.assertRaisesRegex(ContractError, 'device nodes'):
                validate_launch(root, '/probe.py', file_hash(entry), 12345)
            (root / 'dev/nvidia0').unlink()
            with self.assertRaisesRegex(ContractError, 'identity'):
                validate_launch(root, '/probe.py', 'bad', 12345)
            with self.assertRaisesRegex(ContractError, 'nonprivileged'):
                validate_launch(root, '/probe.py', file_hash(entry), 0)
