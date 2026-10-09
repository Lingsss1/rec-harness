import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.tool_shadow import freeze_suite,run_validation


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.config={'protocol_hash':'p','datasets':['D'],'development_seeds':[42],
            'model_versions':{'A':{'sha256':'source-a'},'B':{'sha256':'source-b'}}}
        write_json(self.root/'study/study.json',self.config)
        spec={'study':'study','conditions':[{'id':v,'model_version':v,'seed':42} for v in ['A','B']],
            'identity':{'protocol':'p'},'execution':{'recipe':{},'python':'unused','uid_registry':'unused'},
            'tool_files':['diagnostic.py'],'budget':8,'case_cost':1,'repeat_tolerance':0,'timeout':5,'measurement_bounds':{'mean':[0,1]}}
        write_json(self.root/'spec.json',spec);(self.root/'candidate').mkdir();(self.root/'candidate/diagnostic.py').write_text('# fixture\n')
        self.addCleanup(patch.stopall)
        patch('rec_harness.tool_shadow.load_config',return_value=self.config).start()
        self.audit=patch('rec_harness.tool_shadow.audit_job',side_effect=self.job).start()

    def job(self,root,config,version,dataset,seed):
        return {'metrics':{'HR@10':0.2 if version=='A' else 0.4},'checkpoint_sha256':'host-only-hash'},{'seed':seed,'dataset':dataset}

    def execute(self,directory,recipe,files,*args,**kwargs):
        self.assertEqual(set(files),{'worker.py','request.json','tool/diagnostic.py'})
        request=read_json(files['request.json']['path']);self.assertNotIn('host-only-hash',str(request))
        output=Path(directory)/'results/measurement.json'
        write_json(output,{'measurements':{'mean':request['runs'][0]['metrics']['HR@10']},'limitations':['fixture']})
        return {'process':{'status':'success'},'wall_seconds':0.1,'collection':{'artifacts':{'measurement.json':kwargs['artifacts'].put(output)}}}

    def test_two_audited_conditions_repeat_in_fresh_jobs_and_resume_without_rerun(self):
        freeze_suite(self.root/'spec.json',self.root/'suite')
        with patch('rec_harness.tool_shadow.run_cpu_job',side_effect=self.execute) as worker:
            result=run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json')
            self.assertTrue(result['passed']);self.assertFalse(result['online']);self.assertEqual(result['total_cost'],4)
            self.assertEqual(run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json'),result)
            self.assertEqual(worker.call_count,4)
        self.audit.side_effect=lambda *_:({'metrics':{'HR@10':0.9}},{'seed':42,'dataset':'D'})
        with self.assertRaisesRegex(ContractError,'aggregate differs'):
            run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json')

    def test_alias_conditions_and_confirmation_seeds_are_rejected(self):
        spec=read_json(self.root/'spec.json');spec['conditions'][1]['seed']=17;write_json(self.root/'spec.json',spec)
        with self.assertRaisesRegex(ContractError,'development'):freeze_suite(self.root/'spec.json',self.root/'suite')
        spec['conditions'][1]['seed']=42;spec['conditions'][1]['model_version']='A';write_json(self.root/'spec.json',spec)
        with self.assertRaisesRegex(ContractError,'aliases'):freeze_suite(self.root/'spec.json',self.root/'suite')

    def test_unstable_measurements_cannot_pass_by_repeated_validation(self):
        freeze_suite(self.root/'spec.json',self.root/'suite');calls=[]
        def unstable(directory,recipe,files,*args,**kwargs):
            receipt=self.execute(directory,recipe,files,*args,**kwargs);calls.append(directory)
            output=Path(directory)/'results/measurement.json'
            write_json(output,{'measurements':{'mean':len(calls)/10},'limitations':['unstable fixture']})
            receipt['collection']['artifacts']['measurement.json']=kwargs['artifacts'].put(output);return receipt
        with patch('rec_harness.tool_shadow.run_cpu_job',side_effect=unstable):
            report=run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json')
            self.assertFalse(report['passed']);self.assertEqual(report['gain'],0)
            self.assertEqual(run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json'),report)
            self.assertEqual(len(calls),4)
