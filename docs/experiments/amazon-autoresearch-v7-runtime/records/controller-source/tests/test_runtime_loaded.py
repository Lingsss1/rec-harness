import mmap
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.core import Artifacts, ContractError, digest
from rec_harness.runtime_archive import host_identity
from rec_harness.runtime_loaded import audit_loaded_runtime, mapped_files, mapped_fd_identity, userspace_mappings, mapping_region


class LoadedRuntimeTests(unittest.TestCase):
    def test_gpu_data_mapping_classification_keeps_code_and_other_devices_strict(self):
        from types import SimpleNamespace
        with mmap.mmap(-1,1) as anonymous:
            fields=mapping_region(anonymous)
            maps=' '.join(fields)
            kept,excluded=userspace_mappings(maps,[4])
            self.assertEqual(kept,'');self.assertIn('/dev/zero (deleted)',excluded)
            with self.assertRaisesRegex(ContractError,'deleted'):mapped_files(maps)
            fields[1]='r-xs'
            with self.assertRaisesRegex(ContractError,'protection'):userspace_mappings(' '.join(fields),[4])
        device='1000-2000 rw-s 0 00:06 7 /dev/nvidia4'
        with patch('pathlib.Path.stat',return_value=SimpleNamespace(st_mode=stat.S_IFCHR)):
            kept,excluded=userspace_mappings(device,[4]);self.assertIn('/dev/nvidia4',excluded)
            self.assertEqual(userspace_mappings(device,[5])[0],device)
        with patch('pathlib.Path.stat',return_value=SimpleNamespace(st_mode=stat.S_IFREG)):
            with self.assertRaisesRegex(ContractError,'character device'):userspace_mappings(device,[4])
        deleted='1000-2000 r-xp 0 00:05 8 /tmp/library.so (deleted)'
        kept,_=userspace_mappings(deleted,[4])
        with self.assertRaisesRegex(ContractError,'deleted'):mapped_files(kept)

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

    def test_path_replacement_after_descriptor_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'runtime';source.write_bytes(b'unchanged bytes')
            artifacts=Artifacts(Path(directory)/'objects')
            body={'schema':'gpu-runtime-bytes-v1','host_identity':host_identity(),
                  'files':{str(source).lstrip('/'):artifacts.put(source)}}
            report={**body,'hash':digest(body)}
            with source.open('rb') as handle,mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ):
                maps='\n'.join(line for line in Path('/proc/self/maps').read_text().splitlines() if str(source) in line)
                def replace_after_identity(fd):
                    result=mapped_fd_identity(fd)
                    source.unlink();source.write_bytes(b'unchanged bytes')
                    return result
                with patch('rec_harness.runtime_loaded.mapped_fd_identity',side_effect=replace_after_identity):
                    with self.assertRaisesRegex(ContractError,'changed during audit'):
                        audit_loaded_runtime(report,maps=maps)
