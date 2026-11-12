import tempfile
import unittest
from pathlib import Path
from rec_harness.core import ContractError,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.activation_runtime import run_probe


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        source=self.root/'model.py';source.write_text('# fixture')
        self.config={'source':{'path':str(source),'sha256':file_hash(source)},'budget':2,'cost':1,'timeout':5,
            'settings':{'HIDDEN_UNITS':3,'NUM_BLOCKS':1,'KERNEL_SIZE':3,'DROPOUT_RATE':0.,'MAXLEN':16},
            'execution':{'recipe':{},'python':'unused','uid_registry':'unused'}}

    def runner(self,directory,recipe,files,*args,**kwargs):
        effects={name:{'histories':n,'tail_change_rms':0.,'tail_reference_rms':1.,'relative_tail_change':0.} for name,n in [('all',5),('short',3),('long',2)]}
        report={'schema':'masked-residual-probe-v1','source_hash':files['model.py']['sha256'],'probe_hash':files['activation_probe.py']['sha256'],
            'settings':self.config['settings'],'initialization':'random seed 13, no checkpoint','unchanged_state':True,'scope':'fixture','limitations':[],
            'blocks':[{'block_index':0,'activation_rms':1.,'residual_rms':0.,'zero_fraction':0.,'valid_positions':41,'bypass':effects}]}
        path=directory/'report.json';write_json(path,report)
        return {'process':{'status':'success'},'collection':{'artifacts':{'report.json':kwargs['artifacts'].put(path)}}}

    def test_success_is_cached_and_bound_to_input_and_artifacts(self):
        calls=[]
        def run(*args,**kwargs):calls.append(1);return self.runner(*args,**kwargs)
        result=run_probe(self.config,self.root/'run',runner=run)
        self.assertEqual(result['status'],'success');self.assertEqual(result['budget']['charged_or_reserved'],1)
        self.assertEqual(run_probe(self.config,self.root/'run',runner=run),result);self.assertEqual(len(calls),1)
        bad={**self.config,'cost':2}
        with self.assertRaisesRegex(ContractError,'changed'):run_probe(bad,self.root/'run',runner=run)
        ref=result['artifacts']['measurement.json'];(self.root/'run/objects'/ref['sha256']).write_text('tampered')
        with self.assertRaises(ContractError):run_probe(self.config,self.root/'run',runner=run)

    def test_failure_retains_cost_and_does_not_retry(self):
        def failed(*args,**kwargs):raise ContractError('worker failure')
        result=run_probe(self.config,self.root/'run',runner=failed)
        self.assertEqual(result['status'],'invalid');self.assertEqual(result['budget']['charged_or_reserved'],1)
        self.assertEqual(run_probe(self.config,self.root/'run',runner=self.runner),result)

    def test_interruption_is_retained_and_not_retried_into_success(self):
        def interrupted(*args,**kwargs):raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):run_probe(self.config,self.root/'run',runner=interrupted)
        # A settled interrupted invocation without terminal publication must also stay terminal.
        result=run_probe(self.config,self.root/'run',runner=self.runner)
        self.assertEqual(result['status'],'interrupted')
        self.assertEqual(set(result['artifacts']),{'model.py','activation_probe.py','worker.py'})
        self.assertEqual(result['budget']['charged_or_reserved'],1)
        self.assertEqual(run_probe(self.config,self.root/'run',runner=self.runner),result)

    def test_success_restores_public_projection_without_rerunning(self):
        result=run_probe(self.config,self.root/'run',runner=self.runner)
        (self.root/'run/measurement.json').write_text('{"tampered":true}')
        again=run_probe(self.config,self.root/'run',runner=lambda *a,**kw:self.fail('unexpected execution'))
        self.assertEqual(again,result)
        self.assertEqual(read_json(self.root/'run/measurement.json'),result['measurement'])

    def test_interrupted_worker_receipts_are_retained(self):
        def interrupted(directory,*args,**kwargs):
            write_json(directory/'request.json',{'public_inputs':'fixture'})
            write_json(directory/'process/process.json',{'status':'interrupted'})
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):run_probe(self.config,self.root/'run',runner=interrupted)
        result=run_probe(self.config,self.root/'run',runner=self.runner)
        self.assertIn('job/request.json',result['artifacts'])
        self.assertIn('job/process/process.json',result['artifacts'])
        self.assertEqual(result['status'],'interrupted')
