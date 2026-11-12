import tempfile
import unittest
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, write_json
from rec_harness.runtime_archive import freeze_runtime, restore_runtime


class RuntimeArchiveTests(unittest.TestCase):
    def test_bytes_restore_without_job_inputs_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); template = root / 'template'
            (template / 'runtime').mkdir(parents=True)
            (template / 'runtime/python').write_text('executable fixture')
            (template / 'runtime/python').chmod(0o755)
            (template / 'inputs').mkdir(); (template / 'inputs/data').write_text('excluded fixture')
            write_json(template / 'jail-manifest.json', {'runtime_dirs': ['/runtime'], 'runtime_files': []})
            artifacts = Artifacts(root / 'objects')
            report = freeze_runtime(template, artifacts, root / 'lock.json')
            self.assertEqual(list(report['files']), ['runtime/python'])
            (template / 'runtime/python').write_text('changed host runtime')
            restored = restore_runtime(report, artifacts, root / 'restored')
            self.assertEqual((restored / 'runtime/python').read_text(), 'executable fixture')
            self.assertEqual((restored / 'runtime/python').stat().st_mode & 0o777, 0o555)
            artifact = report['files']['runtime/python']
            (artifacts.root / artifact['sha256']).write_text('corrupt')
            with self.assertRaisesRegex(ContractError, 'corrupt artifact'):
                restore_runtime(report, artifacts, root / 'bad')

    def test_forbidden_path_rejected_even_with_recomputed_hash(self):
        from rec_harness.runtime_archive import host_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['../escape', '/absolute', 'inputs/private', 'dev/nvidia0']:
                report = {'schema': 'gpu-runtime-bytes-v1', 'files': {name: {}},
                          'modes': {name: 0o444}, 'host_identity': host_identity()}
                report['hash'] = digest(report)
                with self.assertRaises(ContractError):
                    restore_runtime(report, Artifacts(root / 'objects'), root / 'restored')

    def test_job_preparation_restores_archive_without_reading_live_runtime(self):
        from unittest.mock import patch
        from rec_harness.isolation import prepare
        from rec_harness.core import read_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); template = root / 'template'
            (template / 'runtime').mkdir(parents=True)
            (template / 'runtime/python').write_text('frozen executable')
            recipe = {'runtime_dirs': ['/runtime'], 'runtime_files': []}
            write_json(template / 'jail-manifest.json', recipe)
            artifacts = Artifacts(root / 'objects')
            report = freeze_runtime(template, artifacts, root / 'lock.json')
            (template / 'runtime/python').unlink()
            # Device creation and ownership require privilege; runtime copying
            # itself is real and must not fall back to any live host path.
            def fake_mknod(path, mode, device):
                Path(path).touch()
            with patch('rec_harness.isolation.os.geteuid', return_value=0), \
                 patch('rec_harness.isolation.os.chown'), \
                 patch('rec_harness.isolation.os.mknod', side_effect=fake_mknod), \
                 patch('rec_harness.isolation.copy_readonly', side_effect=AssertionError('live runtime read')):
                jail = prepare(root / 'jail', **recipe, runtime_archive=report, runtime_artifacts=artifacts)
                self.assertEqual((jail / 'runtime/python').read_text(), 'frozen executable')
                self.assertEqual(read_json(jail / 'jail-manifest.json')['runtime_archive_hash'], report['hash'])
                with self.assertRaisesRegex(ContractError, 'match requested recipe'):
                    prepare(root / 'wrong', ['/other'], [], runtime_archive=report, runtime_artifacts=artifacts)
                self.assertFalse((root / 'wrong').exists())
                with self.assertRaisesRegex(ContractError, 'supplied together'):
                    prepare(root / 'missing', **recipe, runtime_archive=report)
