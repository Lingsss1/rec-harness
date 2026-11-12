import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from rec_harness.core import ContractError, file_hash
from rec_harness.gpu_job import allocate_uid, input_manifest


class JobStagingTests(unittest.TestCase):
    def test_uid_registry_never_reuses_reserved_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('rec_harness.gpu_job.secrets.randbelow', side_effect=[10, 10, 11]), \
                 patch('rec_harness.gpu_job.pwd.getpwuid', side_effect=KeyError):
                self.assertEqual(allocate_uid(directory), 200010)
                self.assertEqual(allocate_uid(directory), 200011)

    def test_inputs_reject_traversal_and_changed_or_linked_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'model.py'; source.write_text('pass')
            ref = {'path': str(source), 'sha256': file_hash(source)}
            self.assertEqual(input_manifest({'model.py': ref})['model.py'], ref)
            for name in ['../model.py', '/model.py']:
                with self.assertRaises(ContractError): input_manifest({name: ref})
            link = Path(directory) / 'link'; link.symlink_to(source)
            with self.assertRaises(ContractError): input_manifest({'model.py': {**ref, 'path': str(link)}})
            source.write_text('changed')
            with self.assertRaises(ContractError): input_manifest({'model.py': ref})
