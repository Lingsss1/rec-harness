import tempfile
import unittest
from pathlib import Path
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.runtime_archive import host_identity
from rec_harness.adapters.amazon_evidence import archive_job


class EvidenceArchiveTests(unittest.TestCase):
    def test_runtime_lock_retains_bytes_without_flat_graph_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'job';runtime=job/'isolated/jail/usr/lib/libtest.so'
            runtime.parent.mkdir(parents=True);runtime.write_bytes(b'frozen library')
            artifacts=Artifacts(root/'objects');ref=artifacts.put(runtime)
            archive={'schema':'gpu-runtime-bytes-v1','files':{'usr/lib/libtest.so':ref},
                'modes':{'usr/lib/libtest.so':0o444},'recipe_hash':'fixture','host_identity':host_identity(),'scope':'fixture'}
            archive['hash']=digest(archive)
            write_json(job/'isolated/runtime-lock.json',archive)
            for name in ['request','completed']:write_json(job/f'isolated/{name}.json',{'runtime_archive_hash':archive['hash']})
            for name in ['isolated/jail/inputs/model.py','isolated/jail/output/scores.json','isolated/jail/usr/lib/unexpected.txt','checkpoint.pt']:
                path=job/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('keep')
            refs=archive_job(job,artifacts)
            self.assertNotIn('isolated/jail/usr/lib/libtest.so',refs)
            self.assertIn('checkpoint.pt',refs);self.assertIn('isolated/jail/usr/lib/unexpected.txt',refs)
            self.assertIn('isolated/jail/inputs/model.py',refs);self.assertIn('isolated/jail/output/scores.json',refs)
            locked=read_json(artifacts.verify(refs['isolated/runtime-lock.json']))
            self.assertEqual(artifacts.verify(locked['files']['usr/lib/libtest.so']).read_bytes(),b'frozen library')
            self.assertEqual(archive_job(job,artifacts),refs)
            runtime.write_bytes(b'mutated library')
            with self.assertRaisesRegex(ContractError,'runtime bytes differ'):archive_job(job,artifacts)

    def test_legacy_job_remains_fully_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'job';job.mkdir();(job/'model.py').write_text('source')
            refs=archive_job(job,Artifacts(root/'objects'))
            self.assertEqual(set(refs),{'model.py'})
