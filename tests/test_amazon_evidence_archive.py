import tempfile
import unittest
from pathlib import Path
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.runtime_archive import host_identity
from rec_harness.adapters.amazon_evidence import archive_job,ingest,restore_job_archive,job_partition,compact_job_archive,resolve_job_archive


class EvidenceArchiveTests(unittest.TestCase):
    def test_partition_comes_from_frozen_split(self):
        config={'development_seeds':[42],'confirmation_seeds':[17,83]}
        self.assertEqual(job_partition(config,{'seed':42}),'development')
        self.assertEqual(job_partition(config,{'seed':17}),'confirmation')
        self.assertEqual(job_partition({}, {'seed':42}),'unknown')
        for result in ({'seed':42,'partition':'confirmation'},{'seed':99},{'seed':True}):
            with self.assertRaises(ContractError):job_partition(config,result)
        for invalid in ({'development_seeds':[42]},
                        {'development_seeds':[42],'confirmation_seeds':[42]},
                        {'development_seeds':[42,42],'confirmation_seeds':[17]},
                        {'development_seeds':[True],'confirmation_seeds':[17]}):
            with self.assertRaises(ContractError):job_partition(invalid,{'seed':42})

    def test_ingest_keeps_split_on_all_evidence_nodes_and_retry_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'model.py';source.write_text('model source')
            config={'protocol_hash':'p','development_seeds':[42],'confirmation_seeds':[17],
                'model_versions':{'base':{'path':str(source),'sha256':file_hash(source)}}}
            write_json(root/'study.json',config)
            write_json(root/'study-lock.json',{'config_hash':digest(config)})
            for index,(seed,status,attempt) in enumerate([(42,'failed','retry'),(42,'success',None),(17,'success',None)]):
                job_id=f'base-tiny-{seed}'
                job=root/'attempt-history'/job_id/attempt if attempt else root/'jobs'/job_id
                result={'job_id':job_id,'model_version':'base','dataset':'tiny','seed':seed,
                    'status':status,'metrics':{},'allocated_gpu_seconds':1,'process':{'completed_at':index}}
                if attempt:result['task_id']=job_id+'-'+attempt
                write_json(job/'completed.json',result)
            self.assertEqual(len(ingest(root)['imported']),3)
            store=Store(root/'memory',read_only=True)
            try:
                graph=store.snapshot();before=store.verify()
            finally:store.close()
            for record in graph['nodes']:
                self.assertEqual(record['attrs']['partition'],
                    'development' if '-42' in record['id'] else 'confirmation')
            experiments={record['id']:record['attrs'] for record in graph['nodes'] if record['type']=='Experiment'}
            self.assertEqual(experiments['base-tiny-42-retry']['independent_run_id'],
                             experiments['base-tiny-42']['independent_run_id'])
            self.assertEqual(ingest(root)['imported'],[])
            store=Store(root/'memory',read_only=True)
            try:self.assertEqual(store.verify(),before);self.assertEqual(store.snapshot(),graph)
            finally:store.close()

    def test_runtime_lock_retains_bytes_without_flat_graph_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'jobs/base-tiny-42';runtime=job/'isolated/jail/usr/lib/libtest.so'
            runtime.parent.mkdir(parents=True);runtime.write_bytes(b'frozen library')
            artifacts=Artifacts(root/'artifacts/objects');ref=artifacts.put(runtime)
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
            restored=root/'restored';result=restore_job_archive(refs,artifacts,restored)
            self.assertEqual(result['runtime_files'],1)
            self.assertEqual((restored/'isolated/jail/usr/lib/libtest.so').read_bytes(),runtime.read_bytes())
            self.assertEqual((restored/'isolated/jail/usr/lib/libtest.so').stat().st_mode&0o777,0o444)
            self.assertEqual((restored/'checkpoint.pt').read_text(),'keep')
            with self.assertRaisesRegex(ContractError,'new destination'):restore_job_archive(refs,artifacts,restored)
            conflict=dict(refs);conflict['isolated/jail/usr/lib/libtest.so']=refs['checkpoint.pt']
            with self.assertRaisesRegex(ContractError,'references disagree'):restore_job_archive(conflict,artifacts,root/'conflict')
            self.assertFalse((root/'conflict').exists())
            source=root/'model.py';source.write_text('source')
            config={'protocol_hash':'p','model_versions':{'base':{'path':str(source),'sha256':file_hash(source)}}}
            write_json(root/'study.json',config);write_json(root/'study-lock.json',{'config_hash':digest(config)})
            write_json(job/'completed.json',{'job_id':'base-tiny-42','model_version':'base','dataset':'tiny','seed':42,
                'status':'success','metrics':{'HR@10':0.5},'allocated_gpu_seconds':1,'process':{'completed_at':1}})
            self.assertEqual(ingest(root)['imported'],['base-tiny-42'])
            store=Store(root/'memory');before=store.verify();graph=store.snapshot();store.close()
            fact=next(n for n in graph['nodes'] if n['type']=='Experiment')
            indexed=resolve_job_archive(fact['attrs']['artifact_refs'],artifacts)
            linked=read_json(artifacts.verify(indexed['isolated/runtime-lock.json']))
            self.assertEqual(linked,archive)
            self.assertNotIn('isolated/jail/usr/lib/libtest.so',fact['attrs']['artifact_refs'])
            compact_restore=root/'compact-restored'
            restore_job_archive(fact['attrs']['artifact_refs'],artifacts,compact_restore)
            self.assertEqual((compact_restore/'isolated/jail/usr/lib/libtest.so').read_bytes(),b'frozen library')
            self.assertEqual(ingest(root)['imported'],[])
            store=Store(root/'memory')
            try:self.assertEqual(store.verify(),before);self.assertEqual(store.snapshot(),graph)
            finally:store.close()
            runtime.write_bytes(b'mutated library')
            with self.assertRaisesRegex(ContractError,'runtime bytes differ'):archive_job(job,artifacts)
            runtime.write_bytes(b'frozen library')
            write_json(job/'isolated/completed.json',{'runtime_archive_hash':'another archive'})
            with self.assertRaisesRegex(ContractError,'binding changed'):archive_job(job,artifacts)
            write_json(job/'isolated/completed.json',{'runtime_archive_hash':archive['hash']})
            artifacts.verify(ref).unlink()
            with self.assertRaisesRegex(ContractError,'unavailable/corrupt'):archive_job(job,artifacts)
            with self.assertRaisesRegex(ContractError,'unavailable/corrupt'):restore_job_archive(refs,artifacts,root/'missing')
            self.assertFalse((root/'missing').exists())

    def test_legacy_job_remains_fully_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'job';job.mkdir();(job/'model.py').write_text('source')
            refs=archive_job(job,Artifacts(root/'objects'))
            self.assertEqual(set(refs),{'model.py'})
            result=restore_job_archive(refs,Artifacts(root/'objects'),root/'restored')
            self.assertEqual(result['runtime_files'],0)
            self.assertEqual((root/'restored/model.py').read_text(),'source')

    def test_large_unlocked_job_keeps_every_file_outside_graph(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'jobs/base-tiny-42';source=root/'model.py';source.write_text('model')
            config={'protocol_hash':'p','development_seeds':[42],'confirmation_seeds':[17],
                'model_versions':{'base':{'path':str(source),'sha256':file_hash(source)}}}
            write_json(root/'study.json',config);write_json(root/'study-lock.json',{'config_hash':digest(config)})
            for index in range(400):
                path=job/'isolated/jail/usr/lib'/('library-'+str(index));path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'preserve all original runtime files')
            write_json(job/'completed.json',{'job_id':'base-tiny-42','model_version':'base','dataset':'tiny','seed':42,
                'status':'success','metrics':{'HR@10':0.5},'allocated_gpu_seconds':1,'process':{'completed_at':1}})
            ingest(root);store=Store(root/'memory',read_only=True)
            try:
                graph=store.snapshot();before=store.verify()
                fact=next(n for n in graph['nodes'] if n['type']=='Experiment')
                self.assertLess(len(json.dumps(fact)),2500)
                self.assertLess(store.db.execute('SELECT max(length(body)) FROM snapshots').fetchone()[0],5000)
            finally:store.close()
            artifacts=Artifacts(root/'artifacts/objects');descriptor=fact['attrs']['artifact_refs']
            refs=resolve_job_archive(descriptor,artifacts)
            self.assertEqual(descriptor['file_count'],401);self.assertEqual(len(refs),401)
            self.assertGreater(len(json.dumps(refs)),50000)
            restored=root/'restored';restore_job_archive(descriptor,artifacts,restored)
            self.assertEqual({str(p.relative_to(restored)) for p in restored.rglob('*') if p.is_file()},set(refs))
            for name in refs:self.assertEqual((restored/name).read_bytes(),(job/name).read_bytes())
            self.assertEqual(ingest(root)['imported'],[])
            store=Store(root/'memory',read_only=True)
            try:self.assertEqual(store.verify(),before);self.assertEqual(store.snapshot(),graph)
            finally:store.close()

    def test_compact_manifest_and_objects_fail_closed_before_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input';source.write_bytes(b'original');artifacts=Artifacts(root/'objects')
            refs={'input':artifacts.put(source)};descriptor=compact_job_archive(refs,artifacts)
            self.assertEqual(resolve_job_archive(descriptor,artifacts),refs)
            for broken in [{**descriptor,'file_count':2},{**descriptor,'file_count':True},
                           {**descriptor,'schema':'unknown'},compact_job_archive({'../escape':refs['input']},artifacts),
                           compact_job_archive({'a//b':refs['input'],'a/b':refs['input']},artifacts)]:
                with self.assertRaises(ContractError):restore_job_archive(broken,artifacts,root/'bad')
                self.assertFalse((root/'bad').exists())
            artifacts.verify(descriptor['manifest_ref']).write_text('changed')
            with self.assertRaisesRegex(ContractError,'unavailable/corrupt'):
                restore_job_archive(descriptor,artifacts,root/'bad')
            self.assertFalse((root/'bad').exists())
            # Legacy maps remain readable, including a file literally named schema.
            self.assertEqual(resolve_job_archive({'schema':refs['input']},artifacts),{'schema':refs['input']})
