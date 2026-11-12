import copy
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,read_json,write_json
from rec_harness.toy_isolated import samples,validate_task,evaluate
from rec_harness.runtime import Research


class SyntheticTaskTests(unittest.TestCase):
    def task(self,family='piecewise-v1'):
        return {'family':family,'training_rows':32,'validation_rows':48,'noise_std':0.1}

    def test_old_samples_exact_and_new_families_deterministic(self):
        rng=random.Random(17);expected=[]
        for _ in range(30):
            a,b=rng.uniform(-1,1),rng.uniform(-1,1)
            expected.append(([a,b],.25*a+.15*b+1.5*a*b))
        self.assertEqual(samples(17,30),expected)
        for family in ['bilinear-v1','piecewise-v1','periodic-v1']:
            task=self.task(family);first=samples(17,30,task)
            self.assertEqual(first,samples(17,30,task))
            self.assertEqual([x for x,y in first],[x for x,y in expected])
            self.assertNotEqual(first,expected)
            clean={**task,'noise_std':0}
            self.assertNotEqual(first,samples(17,30,clean))
        self.assertNotEqual(samples(17,30,self.task('piecewise-v1')),samples(17,30,self.task('periodic-v1')))

    def test_invalid_contracts_rejected_before_worker(self):
        invalid=[{}, {'family':'unknown'}, {**self.task(),'family':'unknown'},
            {**self.task(),'training_rows':True},{**self.task(),'validation_rows':10001},
            {**self.task(),'noise_std':float('nan')},{**self.task(),'noise_std':-1},
            {**self.task(),'noise_std':True},{**self.task(),'noise_std':2}]
        for task in invalid:
            with self.subTest(task=task),self.assertRaises(ContractError):validate_task(task)

    def test_host_scoring_and_validation_targets_never_sent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/'model';model.mkdir();(model/'model.py').write_text('# fixture')
            objects=Artifacts(root/'objects');task=self.task('periodic-v1');seen=[]
            def worker(directory,recipe,files,*args,**kwargs):
                request=read_json(files['request.json']['path']);seen.append(request)
                self.assertEqual(set(request),{'seed','training','queries'})
                self.assertEqual(len(request['training']),32);self.assertEqual(len(request['queries']),48)
                self.assertTrue(all(len(row)==2 and all(isinstance(v,float) for v in row) for row in request['queries']))
                write_json(Path(directory)/'results/predictions.json',{'weights':[0.0],'predictions':[0.0]*48})
                return {'process':{'status':'success'}}
            with patch('rec_harness.toy_isolated.run_cpu_job',side_effect=worker):
                evaluate(model,root/'result.json',17,{'recipe':{},'python':'unused','uid_registry':'unused'},objects,30,task=task)
            self.assertEqual(seen[0],seen[1]);self.assertEqual(len(seen),2)
            targets=[y for x,y in samples(10017,48,task)]
            mse=sum(y*y for y in targets)/48
            self.assertEqual(read_json(root/'result.json')['quality'],1/(1+mse))

    def test_task_contract_requires_host_scoring(self):
        root=Path(__file__).resolve().parents[1]
        config=read_json(root/'examples/toy/campaign.json');config['source']=str(root/'examples/toy')
        config['protocol']['synthetic_task']=self.task()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ContractError,'requires isolated host scoring'):Research.init(temporary,config)


class SyntheticOnlineTests(unittest.TestCase):
    def test_task_family_is_frozen_and_reaches_each_paired_evaluation(self):
        from test_online_validation import OnlineValidationTests,PairedAgent
        from rec_harness.online_validation import freeze_suite,run_validation
        fixture=OnlineValidationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        for stage,family in [('development','piecewise-v1'),('confirmation','periodic-v1')]:
            path=fixture.root/(stage+'.json');config=read_json(path)
            config['protocol']['synthetic_task']={'family':family,'training_rows':32,'validation_rows':48,'noise_std':0.1}
            write_json(path,config)
        suite=freeze_suite(fixture.root/'spec.json',fixture.root/'suite')
        self.assertIn('declared task families',suite['qualification'])
        for stage,family in [('development','piecewise-v1'),('confirmation','periodic-v1')]:
            received=[]
            def evaluate_fixture(model,output,seed,*args,task=None):
                received.append(task)
                return fixture.evaluate(model,output,seed,*args)
            with patch('rec_harness.toy_isolated.evaluate',side_effect=evaluate_fixture):
                report=run_validation(fixture.root/'suite/suite.json',fixture.root/'candidate',fixture.root/stage/'result.json',stage,agent_factory=lambda *_:PairedAgent())
            self.assertFalse(report['online']);self.assertEqual(len(received),4)
            self.assertTrue(all(task['family']==family for task in received))
