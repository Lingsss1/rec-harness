import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.tool_validation import run_validation


class ToolReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'candidate').mkdir();(self.root/'candidate/diagnostic.py').write_text('# candidate fixture\n')
        self.suite={'identity':{'protocol':'fixture'},'tool_files':['diagnostic.py'],'budget':4,'case_cost':1,'timeout':5,
            'execution':{'recipe':{},'python':'unused','uid_registry':'unused'},'cases':[
                {'id':'positive','stage':'correctness','input':{'values':[1,3]},'expected':{'spread':2},'absolute_tolerance':1e-9},
                {'id':'null','stage':'correctness','input':{'values':[4,4]},'expected':{'spread':0},'absolute_tolerance':1e-9}]}
        write_json(self.root/'suite.json',self.suite)

    def worker(self,directory,recipe,files,*args,**kwargs):
        self.assertEqual(set(files),{'request.json','worker.py','tool/diagnostic.py'})
        request=read_json(files['request.json']['path']);self.assertEqual(set(request),{'values'})
        values=request['values'];path=Path(directory)/'results/measurement.json'
        write_json(path,{'measurements':{'spread':max(values)-min(values)},'limitations':['fixture']})
        return {'process':{'status':'success'},'wall_seconds':0.1,'collection':{'artifacts':{'measurement.json':kwargs['artifacts'].put(path)}}}

    def test_host_reference_stays_outside_jail_and_cached_cases_are_not_rerun(self):
        with patch('rec_harness.tool_validation.run_cpu_job',side_effect=self.worker) as worker:
            result=run_validation(self.root/'suite.json',self.root/'candidate',self.root/'result.json','correctness')
            self.assertTrue(result['passed']);self.assertFalse(result['online']);self.assertEqual(result['total_cost'],2)
            self.assertEqual(run_validation(self.root/'suite.json',self.root/'candidate',self.root/'result.json','correctness'),result)
            self.assertEqual(worker.call_count,2)
        (self.root/'candidate/diagnostic.py').write_text('# changed\n')
        with self.assertRaisesRegex(ContractError,'frozen reference validation changed'):
            run_validation(self.root/'suite.json',self.root/'candidate',self.root/'result.json','correctness')

    def test_wrong_measurement_fails_without_retrying_into_a_pass(self):
        self.suite['cases'][0]['expected']['spread']=7;write_json(self.root/'suite.json',self.suite)
        with patch('rec_harness.tool_validation.run_cpu_job',side_effect=self.worker) as worker:
            result=run_validation(self.root/'suite.json',self.root/'candidate',self.root/'result.json','correctness')
            self.assertFalse(result['passed']);self.assertEqual(result['measurement_accuracy'],0.5)
            self.assertEqual(run_validation(self.root/'suite.json',self.root/'candidate',self.root/'result.json','correctness'),result)
            self.assertEqual(worker.call_count,2)
