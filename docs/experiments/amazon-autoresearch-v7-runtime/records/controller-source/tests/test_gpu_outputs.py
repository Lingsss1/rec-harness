import os
from pathlib import Path
import tempfile
import unittest

from rec_harness.core import Artifacts, ContractError, write_json
from rec_harness.gpu_outputs import collect_outputs


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'; self.output.mkdir()
        self.process = self.root / 'process'
        write_json(self.process / 'process.json', {'status': 'success'})
        self.artifacts = Artifacts(self.root / 'objects')

    def collect(self, **limits):
        return collect_outputs(self.output, self.process, self.artifacts, **limits)

    def test_terminal_regular_outputs_are_archived_with_limits(self):
        (self.output / 'result.json').write_bytes(b'{}')
        result = self.collect(max_bytes=2, max_files=1)
        self.assertEqual(result['total_bytes'], 2)
        self.assertEqual(self.artifacts.verify(result['artifacts']['result.json']).read_bytes(), b'{}')
        with self.assertRaisesRegex(ContractError, 'bytes'):
            self.collect(max_bytes=1)
        (self.process / 'process.json').unlink()
        with self.assertRaisesRegex(ContractError, 'terminal'):
            self.collect()

    def test_symlink_hardlink_fifo_directory_and_root_symlink_rejected(self):
        outside = self.root / 'outside'; outside.write_text('synthetic host canary')
        path = self.output / 'bad'
        makers = [lambda: path.symlink_to(outside), lambda: os.link(outside, path),
                  lambda: os.mkfifo(path), lambda: path.mkdir()]
        for make in makers:
            make()
            with self.assertRaises(ContractError): self.collect()
            if path.is_dir() and not path.is_symlink(): path.rmdir()
            else: path.unlink()
        alias = self.root / 'alias'; alias.symlink_to(self.output)
        with self.assertRaisesRegex(ContractError, 'symlink'):
            collect_outputs(alias, self.process, self.artifacts)
