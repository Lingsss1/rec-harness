"""Real host aggregation and model functions, scripted roles and CPU launcher."""
import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,read_json,write_json
from rec_harness.diagnostic_inputs import OBSERVATION_CONTRACT
from rec_harness.runtime import Research
from rec_harness.tool_online import freeze_suite,run_validation
from rec_harness.tool_exposure import audit_exposure
from rec_harness.toy_observations import summarize,freeze_options,from_receipts
from rec_harness.store import Store
import test_online_validation


class OutputObservationTests(unittest.TestCase):
    setUp=test_online_validation.OnlineValidationTests.setUp

    def test_analytic_output_statistics_and_empty_group_are_explicit(self):
        summary=summarize([-2,0,2],[[-1,0],[0,0],[1,0]])
        values=summary['measurements']
        self.assertEqual(values['output.mean'],0)
        self.assertAlmostEqual(values['output.rms'],math.sqrt(8/3))
        self.assertEqual(values['input0.negative.output_mean'],-2)
        self.assertEqual(values['input0.nonnegative.output_mean'],1)
        empty=summarize([2],[[1,0]])
        self.assertEqual(empty['measurements']['input0.negative.count'],0)
        self.assertNotIn('input0.negative.output_mean',empty['measurements'])
        self.assertTrue(any('no samples' in text for text in empty['limitations']))
        with self.assertRaises(ContractError):summarize([float('nan')],[[1,0]])

    def test_invalid_or_changed_producer_rejected_before_campaign_creation(self):
        original=read_json(self.root/'development.json')
        for value in ['yes',None,1]:
            config={**original,'protocol':{**original['protocol'],'output_activation_observations':value}}
            with self.assertRaises(ContractError):Research.init(self.root/'bad',config)
            self.assertFalse((self.root/'bad').exists())
        config={**original,'protocol':{**original['protocol'],'output_activation_observations':True},'execution':{}}
        with self.assertRaises(ContractError):freeze_options(config)
        config['execution']={**original['execution'],'output_observation_producer_hash':'wrong'}
        with self.assertRaises(ContractError):freeze_options(config)

    def test_incomplete_or_nonreproducible_receipts_cannot_generate_observations(self):
        objects=Artifacts(self.root/'objects');path=self.root/'predictions.json'
        write_json(path,{'weights':[0.0],'predictions':[0.0]*120});ref=objects.put(path)
        receipt={'process':{'status':'success'},'collection':{'artifacts':{'predictions.json':ref}}}
        metrics,row=from_receipts({'receipts':[receipt,receipt]},objects,{},101,'root','baseline-0-seed-101')
        self.assertEqual(row['model_version'],'root');self.assertEqual(row['measurements']['output.rms'],0)
        self.assertGreater(metrics['mse'],0)
        for execution in [{'receipts':[receipt]}, {'receipts':[{},{}]}, {'receipts':None}]:
            with self.assertRaises(ContractError):from_receipts(execution,objects,{},101,'root','baseline')
        write_json(path,{'weights':[0.0],'predictions':[1.0]*120})
        different={'process':{'status':'success'},'collection':{'artifacts':{'predictions.json':objects.put(path)}}}
        with self.assertRaisesRegex(ContractError,'differ between processes'):
            from_receipts({'receipts':[receipt,different]},objects,{},101,'root','baseline')

    def test_two_round_three_group_online_observations_follow_incumbent_and_resume(self):
        for stage in ['development','confirmation']:
            path=self.root/(stage+'.json');config=read_json(path)
            config['protocol']['output_activation_observations']=True
            config.update(max_rounds=2,budget=24);write_json(path,config)
        for group in ['existing','candidate']:
            source=self.root/group;source.mkdir(exist_ok=True)
            (source/'diagnostic.py').write_text('def measure(request):\n'
                '    rows=request["observations"]["activation_diagnostics"]\n'
                '    return {"measurements":{"rms":rows[0]["measurements"]["output.rms"]},"limitations":["output summary only"]}\n')
        spec=read_json(self.root/'spec.json');spec.update(existing_tool='existing',tool_files=['diagnostic.py'],
            tool_runtime={'contract':OBSERVATION_CONTRACT,'cost':1,'timeout':10,
                          'execution':{'recipe':{},'python':'unused','uid_registry':'unused'}})
        write_json(self.root/'spec.json',spec);freeze_suite(self.root/'spec.json',self.root/'suite')
        predictions=[];requests=[];visible=[]
        class Agent(test_online_validation.PairedAgent):
            def run(agent,role,context,directory):
                if role=='evidence':visible.append(context)
                return super().run(role,context,directory)
        def train(directory,recipe,files,*args,**kwargs):
            request=read_json(files['request.json']['path'])
            self.assertEqual(set(request),{'seed','training','queries'})
            module_spec=importlib.util.spec_from_file_location('fixture_model',files['model/model.py']['path'])
            model=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(model)
            weights=model.train(request['training'],request['seed'])
            response={'weights':weights,'predictions':[model.predict(weights,row) for row in request['queries']]}
            path=Path(directory)/'results/predictions.json';write_json(path,response);predictions.append(response)
            return {'process':{'status':'success'},'collection':{'artifacts':{'predictions.json':kwargs['artifacts'].put(path)}}}
        def diagnostic(directory,recipe,files,*args,**kwargs):
            from rec_harness.diagnostic_worker import main
            inputs=Path(directory)/'inputs';output=Path(directory)/'output';output.mkdir(parents=True)
            for name,ref in files.items():
                path=inputs/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(Path(ref['path']).read_bytes())
            request=read_json(inputs/'request.json');requests.append(request)
            def keys(value):
                if isinstance(value,dict):return set(value).union(*(keys(v) for v in value.values()))
                if isinstance(value,list):return set().union(*(keys(v) for v in value))
                return set()
            self.assertFalse({'predictions','weights','training','queries'} & keys(request))
            main(inputs,output)
            return {'process':{'status':'success'},'collection':{'artifacts':{'measurement.json':kwargs['artifacts'].put(output/'measurement.json')}}}
        with patch('rec_harness.toy_isolated.run_cpu_job',side_effect=train),patch('rec_harness.diagnostics.run_cpu_job',side_effect=diagnostic):
            reports=[]
            for stage in ['utility','confirmation']:
                output=self.root/stage/'result.json'
                report=run_validation(self.root/'suite/suite.json',self.root/'candidate',output,stage,agent_factory=lambda *_:Agent())
                reports.append(report)
                before=(len(predictions),len(requests),len(visible))
                self.assertEqual(run_validation(self.root/'suite/suite.json',self.root/'candidate',output,stage,agent_factory=lambda *_:Agent()),report)
                self.assertEqual((len(predictions),len(requests),len(visible)),before)
        self.assertTrue(all(report['passed'] for report in reports));self.assertTrue(all(not report['online'] for report in reports))
        self.assertEqual(len(requests),8)
        self.assertEqual({request['model_version'] for request in requests},{'root','attempt-1'})
        for context in visible:
            self.assertEqual(context['activation_diagnostics'][0]['model_version'],context['model_version'])
            self.assertIn('Scalar output',context['activation_diagnostics'][0]['scope'])
        for report in reports:
            self.assertEqual(report['gain'],0)
            for group in report['comparison_groups']:
                campaign=Path(report['rows'][0][group]['graph_directory']).parent
                audit=audit_exposure(campaign)
                self.assertEqual(len(audit['exposures']),0 if group=='no_tool' else 2)
                store=Store(campaign/'memory',read_only=True)
                try:
                    nodes=[n for n in store.snapshot()['nodes'] if n['id'].endswith('-output-activation')]
                    self.assertEqual(len(nodes),3)
                finally:store.close()
