"""Paired controller tests; injected agents explicitly cannot report online evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.online_validation import freeze_suite,load_suite,run_validation
from test_runtime import ROOT,ScriptedAgent


class PairedAgent(ScriptedAgent):
    def run(self,role,context,directory):
        response=super().run(role,context,directory)
        if role=='hypothesis':response['proposal']['controls']=context['supported_controls']
        return response


class OnlineValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        write_json(self.root/'template/jail-manifest.json',{})
        cases=[]
        for stage,seed in [('development',101),('confirmation',202)]:
            cfg=read_json(ROOT/'examples/toy/campaign.json');cfg.update(source=str(ROOT/'examples/toy'),max_rounds=1,budget=12)
            cfg.setdefault('codex',{})['model']='fixture-model'
            cfg['protocol']['seeds']=[seed];cfg['protocol']['guardrails']={'serialized_coefficients':16}
            cfg['agent_backend']={'kind':'jailed_codex','template':str(self.root/'template'),'runtime_root':'unused'}
            cfg['execution']={'kind':'isolated-toy-v1','python':'unused','recipe':{},'uid_registry':'unused'}
            write_json(self.root/(stage+'.json'),cfg);cases.append({'id':stage,'stage':stage,'config':stage+'.json'})
        write_json(self.root/'spec.json',{'cases':cases});write_json(self.root/'candidate/policy.json',{'prompts':{'evidence':'Check alternatives.'}})

    def evaluate(self,model,output,seed,*args):
        write_json(output,{'quality':0.9 if 'attempt-1' in str(output) else 0.5,'serialized_coefficients':4})
        return {'status':'success','wall_seconds':0,'gpu_seconds':0}

    def test_real_controllers_pair_isolated_graphs_resume_and_refuse_mock_online_label(self):
        freeze_suite(self.root/'spec.json',self.root/'suite')
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.evaluate):
            report=run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json','development',agent_factory=lambda *_:PairedAgent())
            self.assertEqual(report['evaluation_mode'],'end_to_end_candidate_generation');self.assertFalse(report['online']);self.assertEqual(report['gain'],0);self.assertEqual(report['total_cost'],16)
            pair=report['rows'][0];self.assertNotEqual(pair['baseline']['graph_directory'],pair['candidate']['graph_directory'])
            self.assertEqual(pair['baseline']['agent_calls'],6)
            again=run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json','development',agent_factory=lambda *_:PairedAgent())
            self.assertEqual(again,report)
        write_json(self.root/'candidate/policy.json',{})
        with self.assertRaisesRegex(ContractError,'inputs changed'):
            run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json','development')

    def test_suite_rejects_reused_conditions_and_source_corruption(self):
        cfg=read_json(self.root/'confirmation.json');cfg['protocol']['seeds']=[101];write_json(self.root/'confirmation.json',cfg)
        with self.assertRaisesRegex(ContractError,'reuse'):freeze_suite(self.root/'spec.json',self.root/'bad-suite')
        cfg['protocol']['data_version']='renamed-data';cfg['protocol']['seeds']=[10101];write_json(self.root/'confirmation.json',cfg)
        with self.assertRaisesRegex(ContractError,'reuse'):freeze_suite(self.root/'spec.json',self.root/'renamed-suite')
        cfg['protocol']['seeds']=[202];write_json(self.root/'confirmation.json',cfg)
        suite=freeze_suite(self.root/'spec.json',self.root/'suite')
        ref=suite['cases'][0]['inputs']['model.py'];(self.root/'suite/objects'/ref['sha256']).write_text('changed')
        with self.assertRaises(ContractError):load_suite(self.root/'suite/suite.json')

    def test_new_suite_cannot_depend_on_mutable_default_model(self):
        cfg=read_json(self.root/'development.json');cfg['codex'].pop('model');write_json(self.root/'development.json',cfg)
        with self.assertRaisesRegex(ContractError,'explicit codex.model'):freeze_suite(self.root/'spec.json',self.root/'suite')

    def test_reference_suite_cannot_claim_empty_graph_with_external_memory(self):
        for key in ['initial_memory_files','initial_memory']:
            cfg=read_json(self.root/'development.json');cfg[key]=['mutable-memory.json']
            write_json(self.root/'development.json',cfg)
            with self.assertRaisesRegex(ContractError,'unmodified reference'):
                freeze_suite(self.root/'spec.json',self.root/key)
            cfg.pop(key);write_json(self.root/'development.json',cfg)
