import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from rec_harness.adapters.amazon_study import execute_job
from rec_harness.adapters.contract_gate import run_gate
from rec_harness.core import file_hash, write_json, digest


class TrainingGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['source.py', 'worker.py', 'sample_train.txt', 'sample_valid.txt']:
            (self.root / name).write_text('fixture')
        self.config = {'python': 'unused', 'model_versions': {'model': {'path': str(self.root / 'source.py'),
                       'sha256': file_hash(self.root / 'source.py')}},
            'worker': str(self.root / 'worker.py'), 'worker_sha256': file_hash(self.root / 'worker.py'),
            'data': str(self.root), 'data_hashes': {name: file_hash(self.root / name) for name in ['sample_train.txt', 'sample_valid.txt']},
            'protocol_hash': 'p', 'job_timeout_seconds': 30,
            'protocol': {key: 1 for key in ['epochs', 'batches', 'eval_users', 'eval_seed', 'patience', 'eval_every']}}

    def train(self, argv, root, output, **kwargs):
        directory = Path(output).parent
        (directory / 'checkpoint.pt').write_bytes(b'fixture checkpoint')
        write_json(directory / 'result.json', {'status': 'success', 'metrics': {'HR@10': 0.5},
                   'checkpoint_sha256': file_hash(directory / 'checkpoint.pt')})
        return {'status': 'success', 'completed_at': 1}

    def test_pre_failure_never_launches_training_and_has_recoverable_receipt(self):
        with patch('rec_harness.adapters.amazon_study.run_gate', return_value={'status': 'invalid'}), \
             patch('rec_harness.adapters.amazon_study.run_process') as training:
            result = execute_job(self.config, self.root, 'model', 'sample', 42, 0)
        training.assert_not_called()
        self.assertEqual(result['status'], 'contract_failure')
        self.assertIsNone(result['metrics'])
        self.assertIn('completed_at', result['process'])
        self.assertGreater(result['allocated_gpu_seconds'], 0)

    def test_post_failure_invalidates_metrics_and_uses_shared_deadline(self):
        with patch('rec_harness.adapters.amazon_study.run_gate', side_effect=[{'status': 'success'}, {'status': 'invalid'}]) as gate, \
             patch('rec_harness.adapters.amazon_study.run_process', side_effect=self.train) as training:
            result = execute_job(self.config, self.root, 'model', 'sample', 42, 0)
        self.assertEqual(result['status'], 'contract_failure')
        self.assertIsNone(result['metrics'])
        self.assertLess(training.call_args.kwargs['timeout'], 30)
        self.assertLess(gate.call_args_list[1].args[-1], gate.call_args_list[0].args[-1])
        self.assertIn('checkpoint_hash', gate.call_args.kwargs)

    def test_success_requires_both_gates_and_training_receipt(self):
        with patch('rec_harness.adapters.amazon_study.run_gate', return_value={'status': 'success'}) as gate, \
             patch('rec_harness.adapters.amazon_study.run_process', side_effect=self.train):
            result = execute_job(self.config, self.root, 'model', 'sample', 42, 0)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(gate.call_count, 2)

    def test_isolated_training_receives_registered_runtime_archive(self):
        archive={'hash':'frozen-runtime-identity'}
        self.config['execution']={'execution_id':'frozen-execution','recipe':{},'runtime_archive':archive}
        def isolated(directory,recipe,files,entry,arguments,**kwargs):
            self.assertEqual(kwargs['runtime_archive'],archive)
            output=self.root/'fixture-output';output.mkdir()
            (output/'checkpoint.pt').write_bytes(b'fixture checkpoint')
            write_json(output/'result.json',{'status':'success','metrics':{'HR@10':0.5},
                'checkpoint_sha256':file_hash(output/'checkpoint.pt')})
            refs=kwargs['artifacts'].snapshot(output,['result.json','checkpoint.pt'])
            return {'process':{'status':'success','completed_at':1},'collection':{'artifacts':refs}}
        with patch('rec_harness.adapters.amazon_study.run_gate',return_value={'status':'success'}), \
             patch('rec_harness.gpu_job.run_reviewed_job',side_effect=isolated) as runner:
            result=execute_job(self.config,self.root,'model','sample',42,0)
        self.assertEqual(result['status'],'success');self.assertEqual(runner.call_count,1)
        self.assertEqual(result['execution_id'],'frozen-execution')

    def test_current_gate_rejects_missing_or_invalid_graph_even_with_valid_report_hash(self):
        from test_tensor_graph import report_fixture,semantics_fixture
        source=self.config['model_versions']['model']
        for variant in ['missing','invalid','valid']:
            def child(argv,cwd,directory,**kwargs):
                report=report_fixture();report['model_id']='model';report['source_hash']=source['sha256']
                graph=report['tensor_graph'];graph.update(model_id='model',source_hash=source['sha256'])
                if variant=='invalid':graph['operations'][0]['inputs']=['missing-tensor']
                graph['hash']=digest({k:v for k,v in graph.items() if k!='hash'})
                if variant=='missing':report.pop('tensor_graph')
                report.update(inspector_hash=file_hash(argv[1]),checks={k:True for k in
                    ['causal_input_gradient','tail_readout','empty_history_finite','padding_gradient_zero']})
                report['checks']['sequence_semantics']=semantics_fixture()
                report['hash']=digest(report)
                write_json(Path(argv[argv.index('--output')+1]),report)
                return {'status':'success'}
            with self.subTest(variant=variant),patch('rec_harness.adapters.contract_gate.run_process',side_effect=child):
                result=run_gate('unused',source,'model',self.root/variant,{},30)
            self.assertEqual(result['status'],'success' if variant=='valid' else 'invalid')
            if variant=='valid':self.assertEqual(result['tensor_graph']['operation_count'],1)

    def test_relative_gate_paths_survive_child_working_directory_change(self):
        relative=Path(os.path.relpath(self.root,Path.cwd()))
        source={'path':str(relative/'source.py'),'sha256':file_hash(self.root/'source.py')}
        def child(argv,cwd,directory,**kwargs):
            for flag in ['--source','--output','--checkpoint']:
                path=Path(argv[argv.index(flag)+1])
                self.assertTrue(path.is_absolute())
                self.assertTrue(path.parent.exists())
            self.assertTrue(Path(argv[1]).is_file())
            self.assertNotEqual(Path(cwd),Path.cwd()/relative)
            return {'status':'crash'}
        with patch('rec_harness.adapters.contract_gate.run_process',side_effect=child):
            result=run_gate('unused',source,'model',relative/'probe',{},30,
                            checkpoint=relative/'source.py',checkpoint_hash=source['sha256'])
        self.assertEqual(result['status'],'crash')
