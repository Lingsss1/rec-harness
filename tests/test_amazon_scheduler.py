from pathlib import Path
import tempfile
import unittest
from threading import Event
from unittest.mock import patch
from rec_harness.adapters.amazon_study import run_study
from rec_harness.core import digest,read_json,write_json
from rec_harness.store import Store

class AmazonSchedulerTests(unittest.TestCase):
    def test_refill_real_study_ledger_then_resume_without_reexecution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);released=Event();devices={}
            config={'model_versions':{'m':{}},'development_seeds':[42],'confirmation_seeds':[17],
                'datasets':['slow','fast','next'],'gpu_ids':[0,1],'job_timeout_seconds':10,'protocol_hash':'p'}
            write_json(root/'study.json',config);write_json(root/'study-lock.json',{'config_hash':digest(config)})
            store=Store(root/'memory');store.configure_budget(25);store.close()
            def execute(config,root,model,dataset,seed,gpu,task_id):
                devices[dataset]=gpu
                if dataset=='slow' and not released.wait(3):raise AssertionError('new job waited for slow batch peer')
                if dataset=='next':released.set()
                job=f'{model}-{dataset}-{seed}'
                result={'job_id':job,'task_id':task_id,'model_version':model,'dataset':dataset,'seed':seed,
                    'status':'success','metrics':{'HR@10':.5},'allocated_gpu_seconds':1}
                write_json(root/'jobs'/job/'completed.json',result);return result
            with patch('rec_harness.adapters.amazon_study.execute_job',side_effect=execute):
                report=run_study(root)
            self.assertEqual(report['budget']['charged_or_reserved'],3)
            self.assertEqual(devices['next'],devices['fast'])
            self.assertEqual(read_json(root/'status.json')['status'],'completed')
            store=Store(root/'memory');before=store.verify();store.close()
            with patch('rec_harness.adapters.amazon_study.execute_job',side_effect=AssertionError('resume retrained')):
                self.assertEqual(run_study(root)['budget'],report['budget'])
            store=Store(root/'memory');self.assertEqual(store.verify(),before);store.close()

    def test_failure_retains_reservation_and_prevents_unaffordable_next_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);started=[]
            config={'model_versions':{'m':{}},'development_seeds':[42],'confirmation_seeds':[17],
                'datasets':['bad','good','pending'],'gpu_ids':[0],'job_timeout_seconds':10,'protocol_hash':'p'}
            write_json(root/'study.json',config);write_json(root/'study-lock.json',{'config_hash':digest(config)})
            store=Store(root/'memory');store.configure_budget(21);store.close()
            def execute(config,root,model,dataset,seed,gpu,task_id):
                started.append(dataset)
                if dataset=='bad':raise RuntimeError('receipt unavailable')
                job=f'{model}-{dataset}-{seed}'
                result={'job_id':job,'task_id':task_id,'model_version':model,'dataset':dataset,'seed':seed,
                    'status':'success','metrics':{'HR@10':.5},'allocated_gpu_seconds':2}
                write_json(root/'jobs'/job/'completed.json',result);return result
            with patch('rec_harness.adapters.amazon_study.execute_job',side_effect=execute):
                report=run_study(root)
            self.assertEqual(started,['bad','good'])
            self.assertEqual(report['budget']['charged_or_reserved'],12)
            self.assertEqual(read_json(root/'status.json')['status'],'budget_stopped')
            store=Store(root/'memory')
            try:
                settled=[e['payload'] for e in store.events() if e['kind']=='budget_settled']
                self.assertEqual(settled[0]['status'],'invalid')
                self.assertIsNone(settled[0]['actual'])
                row=store.db.execute("SELECT reserved, actual FROM budget WHERE id='m-bad-42'").fetchone()
                self.assertEqual(row['reserved'],10);self.assertIsNone(row['actual'])
                self.assertTrue(any(e['kind']=='study_budget_stop' for e in store.events()))
                store.verify()
            finally:store.close()
