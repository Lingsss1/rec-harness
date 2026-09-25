import tempfile
import unittest
from pathlib import Path

from rec_harness.core import ContractError, file_hash, write_json
from rec_harness.gpu_launcher import validate_launch


class LaunchValidationTests(unittest.TestCase):
    def test_identity_and_path_fail_before_cuda_or_code_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'jail-manifest.json', {'gpu_devices': [0], 'uid': 65534})
            script = root / 'worker.py'; script.write_text('raise RuntimeError("must not execute")\n')
            for device, uid, path, sha, message in [
                (1, 65534, '/worker.py', file_hash(script), 'differs'),
                (0, 0, '/worker.py', file_hash(script), 'nonprivileged'),
                (0, 65534, '/inputs/../worker.py', file_hash(script), 'absolute'),
                (0, 65534, '/worker.py', 'incorrect', 'hash mismatch')]:
                with self.assertRaisesRegex(ContractError, message):
                    validate_launch(root, path, sha, device, uid)

    def test_entry_symlink_cannot_escape_jail_during_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'jail'; root.mkdir()
            outside = Path(directory) / 'outside.py'; outside.write_text('pass')
            write_json(root / 'jail-manifest.json', {'gpu_devices': [0], 'uid': 65534})
            (root / 'worker.py').symlink_to(outside)
            with self.assertRaises(ContractError):
                validate_launch(root, '/worker.py', file_hash(outside), 0, 65534)
