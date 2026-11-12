from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rec_harness.adapters.amazon_execution import register_execution, resolve_execution, preflight_execution,inspect_execution
from rec_harness.core import Artifacts, ContractError, write_json
from rec_harness.store import Store


class ExecutionRegistrationTests(unittest.TestCase):
    def test_preflight_keeps_gpu_ledger_unchanged_and_rejects_wrong_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store(Path(directory)/'memory');store.configure_budget(100)
            execution={'execution_id':'execution-test','runtime_archive':{'hash':'frozen'}}
            try:
                for status,binding,process_status in [('invalid','frozen','crash'),
                        ('success','other','success'),('success','frozen','success')]:
                    def run(argv,cwd,output,**kwargs):
                        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                        write_json(argv[-1],{'status':status,'runtime_archive_hash':binding,
                            'audit':{'runtime_archive_hash':binding}})
                        return {'status':process_status,'wall_seconds':0.1}
                    before=store.budget()
                    with patch('rec_harness.agents.run_process',side_effect=run):
                        if status=='success' and binding=='frozen':
                            self.assertEqual(preflight_execution(store,execution,'python')['status'],'success')
                        else:
                            with self.assertRaisesRegex(ContractError,'before GPU reservation'):
                                preflight_execution(store,execution,'python')
                    self.assertEqual(store.budget(),before)
                self.assertEqual(len([e for e in store.events() if e['kind']=='amazon_runtime_preflight']),3)
                with patch('rec_harness.agents.run_process') as runner:
                    self.assertIsNone(preflight_execution(store,{'execution_id':'legacy'},'python'))
                    runner.assert_not_called()
            finally:store.close()

    def test_registered_runtime_is_frozen_and_verified_on_resolution(self):
        from rec_harness.runtime_archive import restore_runtime
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);template=root/'template'
            (template/'runtime').mkdir(parents=True)
            (template/'runtime/python').write_text('frozen interpreter')
            (template/'lib').mkdir();(template/'lib/runtime.so').write_text('frozen library')
            (template/'inputs').mkdir();(template/'inputs/private').write_text('excluded data')
            recipe={'runtime_dirs':['/runtime'],'runtime_files':['/lib/runtime.so']}
            write_json(template/'jail-manifest.json',recipe)
            store=Store(root/'memory')
            try:
                first=register_execution(store,recipe,runtime_template=template)
                count=len(store.events())
                self.assertEqual(first,register_execution(store,recipe,runtime_template=template))
                self.assertEqual(len(store.events()),count)
                (template/'runtime/python').write_text('changed live interpreter')
                resolved=resolve_execution(store,first['execution_id'])
                archive=resolved['runtime_archive'];self.assertNotIn('inputs/private',archive['files'])
                artifacts=Artifacts(root/'artifacts/objects')
                restored=restore_runtime(archive,artifacts,root/'restored')
                self.assertEqual((restored/'runtime/python').read_text(),'frozen interpreter')
                with patch('rec_harness.runtime_archive.host_identity',return_value={'different':'host'}):
                    with self.assertRaisesRegex(ContractError,'kernel or driver'):
                        resolve_execution(store,first['execution_id'])
                artifacts.verify(archive['files']['runtime/python']).write_text('corrupt archive')
                with self.assertRaisesRegex(ContractError,'corrupt artifact'):
                    resolve_execution(store,first['execution_id'])
            finally:store.close()

    def test_append_only_idempotence_and_changed_code_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'memory')
            try:
                recipe = {'runtime_dirs': ['/runtime'], 'runtime_files': ['/lib/runtime.so']}
                first = register_execution(store, recipe)
                count = len(store.events())
                self.assertEqual(first, register_execution(store, recipe))
                self.assertEqual(len(store.events()), count)
                self.assertEqual(first, resolve_execution(store, first['execution_id']))
                self.assertEqual(set(first['code_artifacts']), set(first['code_hashes']))
                with patch('rec_harness.adapters.amazon_execution.code_identity', return_value={'changed': 'hash'}):
                    self.assertEqual(inspect_execution(store,first['execution_id']),first)
                    with self.assertRaisesRegex(ContractError, 'code changed'):
                        resolve_execution(store, first['execution_id'])
                with self.assertRaisesRegex(ContractError, 'uniquely registered'):
                    resolve_execution(store, 'unknown')
                artifact = next(iter(first['code_artifacts'].values()))
                (Path(directory) / 'artifacts/objects' / artifact['sha256']).write_text('corrupt')
                with self.assertRaisesRegex(ContractError, 'corrupt artifact'):
                    resolve_execution(store, first['execution_id'])
            finally: store.close()
