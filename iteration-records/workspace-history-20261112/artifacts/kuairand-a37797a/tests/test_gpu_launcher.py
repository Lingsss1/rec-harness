import tempfile
import unittest
import ctypes.util
import importlib.util
import subprocess
import sys
from pathlib import Path

from rec_harness.core import ContractError, file_hash, write_json
from rec_harness.gpu_launcher import validate_launch


class LaunchValidationTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and importlib.util.find_spec('torch')
                         and ctypes.util.find_library('seccomp'), 'requires Linux, torch and seccomp')
    def test_cpu_initialization_after_thread_filter(self):
        code = '''
import os, socket
import torch
from rec_harness.gpu_launcher import warm_cpu_threads
from rec_harness.policy_sandbox import restrict_syscalls
torch.set_num_threads(2)
before = torch.get_rng_state().clone()
warm_cpu_threads(torch)
assert torch.equal(before, torch.get_rng_state())
restrict_syscalls(sync_threads=True)
value = torch.zeros((65536, 48))
assert value.sum().item() == 0
for action in (socket.socket, os.fork):
    try:
        result = action()
    except PermissionError:
        continue
    if isinstance(result, int):
        if result == 0: os._exit(0)
        os.waitpid(result, 0)
    else: result.close()
    raise AssertionError('sandbox operation permitted')
print('cpu-initialization-and-boundary-passed')
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('cpu-initialization-and-boundary-passed', result.stdout)

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
