import mmap
import os
import tempfile
import unittest
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest
from rec_harness.runtime_archive import host_identity
from rec_harness.runtime_loaded import audit_loaded_runtime, mapped_files


class LoadedRuntimeTests(unittest.TestCase):
    def test_live_mapping_rejects_changed_and_unarchived_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'mapped file'; source.write_bytes(b'original runtime')
            artifacts = Artifacts(Path(directory) / 'objects')
            body = {'schema': 'gpu-runtime-bytes-v1', 'host_identity': host_identity(),
                    'files': {str(source).lstrip('/'): artifacts.put(source)}}
            report = {**body, 'hash': digest(body)}
            with source.open('r+b') as handle, mmap.mmap(handle.fileno(), 0) as mapping:
                maps = '\n'.join(line for line in Path('/proc/self/maps').read_text().splitlines()
                                 if str(source) in line)
                self.assertTrue(maps)
                # Sandboxed stat may virtualize st_dev while proc exposes the
                # host device. Normalize this fixture only; production rejects it.
                device = source.stat().st_dev
                normalized = []
                for line in maps.splitlines():
                    fields = line.split(None, 5)
                    fields[3] = f'{os.major(device):x}:{os.minor(device):x}'
                    normalized.append(' '.join(fields))
                maps = '\n'.join(normalized)
                result = audit_loaded_runtime(report, maps=maps)
                self.assertEqual(list(result['files']), [str(source)])
                fields = maps.splitlines()[0].split(None, 5)
                fields[4] = str(int(fields[4]) + 1)
                with self.assertRaisesRegex(ContractError, 'inode changed'):
                    audit_loaded_runtime(report, maps=' '.join(fields))
                mapping[0:1] = b'X'; mapping.flush()
                with self.assertRaisesRegex(ContractError, 'bytes differ'):
                    audit_loaded_runtime(report, maps=maps)
                empty = {**body, 'files': {}}
                with self.assertRaisesRegex(ContractError, 'not archived'):
                    audit_loaded_runtime({**empty, 'hash': digest(empty)}, maps=maps)
                source.unlink(); source.write_bytes(b'original runtime')
                with self.assertRaisesRegex(ContractError, 'inode changed'):
                    audit_loaded_runtime(report, maps=maps)

    def test_maps_handles_escaped_names_and_rejects_deleted_mapping(self):
        self.assertEqual(mapped_files('1000-2000 r-xp 0 08:01 5 /a\\040b'), {'/a b': ('08:01', 5)})
        with self.assertRaisesRegex(ContractError, 'deleted'):
            mapped_files('1000-2000 r-xp 0 08:01 5 /a (deleted)')
        self.assertEqual(mapped_files('1000-2000 rw-p 0 00:00 0 [heap]'), {})
