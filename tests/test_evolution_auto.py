"""Orchestration fixtures, not measured evidence of online research utility."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.autonomous import autoresearch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.evolution import Evolution,STAGES,validate_policy
from rec_harness.store import Store

class Optimizer:
    calls=0
    def run(self,role,context,directory):
        self.calls+=1;Path(directory).mkdir(parents=True,exist_ok=True)
        return {'patch_id':'policy','failure_pattern':'fixture','trace_refs':['trace'],
            'causal_hypothesis':'fixture','editable_scope':['retrieval_limit'],
            'patch_diff':json.dumps({'retrieval_limit':25}),'expected_decision_change':'fixture',
            'regression_risks':['fixture'],'validation_plan':'frozen','rollback_version':'base'}

class EvolutionAutomationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.campaign=self.root/'campaign';self.agent=Optimizer()

    def config(self,reject=None):
        identity={key:'synthetic-fixture' for key in ['base_llm','tools','start','graph','budget','protocol']}
        stages={}
        for stage in STAGES['harness']:
            script=self.root/(stage+'.py')
            report={'passed':stage!=reject,'gain':1,'regression':0,'total_cost':4,
                    'evaluation_ids':[stage],'identity':identity,'comparison_groups':['baseline','candidate'],'online':True}
            script.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('+repr(json.dumps(report))+')\n')
            stages[stage]={'command':['{python}',str(script),'{output}'],'timeout':5,'cost':1,'minimum_gain':0.1,'maximum_regression':0,'maximum_total_cost':8}
        config={'adapter':'research-evolution-v1','candidate_id':'candidate','kind':'harness','context':{'qualification':'unit fixture'},
                'plan':{'budget':12,'agent_cost':1,'development_ids':['source'],'identity':identity,'stages':stages,'rollback_version':'base'}}
        path=self.root/'config.json';write_json(path,config);return path

    def test_interruption_after_committed_stage_resumes_without_repeating_generation_or_validation(self):
        original=Evolution.validate
        def interrupt(instance,candidate_id,stage):
            result=original(instance,candidate_id,stage)
            if stage=='prefix_audit':raise KeyboardInterrupt()
            return result
        with patch.object(Evolution,'validate',interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(),agent=self.agent)
        self.assertEqual(self.agent.calls,1)
        result=autoresearch(self.campaign,agent=self.agent)
        self.assertEqual(result['status'],'promoted');self.assertEqual(result['budget']['charged_or_reserved'],4)
        self.assertEqual(self.agent.calls,1)
        store=Store(self.campaign/'memory');before=store.verify();store.close()
        self.assertEqual(autoresearch(self.campaign,agent=self.agent),result)
        store=Store(self.campaign/'memory');self.assertEqual(store.verify(),before);store.close()

    def test_rejected_scientific_validation_cannot_be_retried_into_promotion(self):
        result=autoresearch(self.campaign,self.config(reject='development'),agent=self.agent)
        self.assertEqual(result['status'],'rejected');self.assertEqual(result['budget']['charged_or_reserved'],3)
        self.assertFalse((self.campaign/'active-harness.json').exists())
        self.assertEqual(autoresearch(self.campaign,agent=self.agent),result)
        self.assertEqual(self.agent.calls,1)

    def test_committed_validation_restores_missing_marker_but_rejects_changed_marker(self):
        autoresearch(self.campaign,self.config(),agent=self.agent)
        evolution=Evolution(self.campaign);self.addCleanup(evolution.store.close)
        marker=self.campaign/'candidates/candidate/validation/prefix_audit/accepted.json'
        expected=read_json(marker);marker.unlink()
        evolution.verify();self.assertEqual(read_json(marker),expected)
        write_json(marker,{**expected,'plan_hash':'forged'})
        with self.assertRaisesRegex(ContractError,'projection differs'):evolution.verify()

    def test_all_tool_stages_are_executed_and_cached(self):
        path=self.config();config=read_json(path);config['kind']='tool';stages={}
        for stage in STAGES['tool']:
            report={'passed':True,'gain':1,'regression':0,'total_cost':7,
                'evaluation_ids':[stage+'-one',stage+'-two'],'identity':config['plan']['identity'],
                'comparison_groups':['no_tool','existing_tool','candidate_tool'],'online':True}
            script=self.root/(stage+'.py')
            script.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('+repr(json.dumps(report))+')\n')
            stages[stage]={'command':['{python}',str(script),'{output}'],'timeout':5,'cost':1,
                'minimum_gain':0.1,'maximum_regression':0,'maximum_total_cost':8}
        config['plan']['stages']=stages;config['plan']['tool_contract']='public-aggregate-measurement-v1';write_json(path,config)
        class ToolAgent:
            calls=0
            def run(self,role,context,directory):
                self.calls+=1;assert role=='tool'
                target=Path(directory)/'model';target.mkdir(parents=True)
                (target/'diagnostic.py').write_text('def measure(values): return sum(values)\n')
                return {'changed_files':['diagnostic.py']}
        agent=ToolAgent();result=autoresearch(self.campaign,path,agent=agent)
        self.assertEqual(result['status'],'promoted');self.assertEqual(result['budget']['charged_or_reserved'],7)
        self.assertEqual(autoresearch(self.campaign,agent=agent),result);self.assertEqual(agent.calls,1)
        from rec_harness.diagnostics import freeze_tool
        from rec_harness.core import Artifacts
        artifacts=Artifacts(self.root/'new-campaign/objects')
        frozen=freeze_tool({'registry':self.campaign,'identity':config['plan']['identity'],
            'execution':{'recipe':{},'python':'python3','uid_registry':'unused'},'cost':1,'timeout':5},artifacts)
        self.assertEqual(frozen['candidate_id'],'candidate')
        for ref in frozen['implementation'].values():artifacts.verify(ref)

    def test_promoted_prompt_reaches_jailed_role_prompt(self):
        from rec_harness.jailed_agent import JailedCodexAgent
        write_json(self.root/'template/jail-manifest.json',{})
        agent=JailedCodexAgent(self.root/'template',self.root/'runtime',prompts={'evidence':'Use the exact frozen base model identity.'})
        self.assertIn('Use the exact frozen base model identity.',agent.role_prompt('evidence'))
        self.assertNotIn('Use the exact frozen base model identity.',agent.role_prompt('lead'))
        with self.assertRaisesRegex(ContractError,'unknown role'):validate_policy({'prompts':{'evidnce':'typo'}})

    def test_first_promotion_can_roll_back_to_builtin_for_new_campaign(self):
        from rec_harness.harness_activation import load_active_policy
        autoresearch(self.campaign,self.config(),agent=self.agent)
        evolution=Evolution(self.campaign);self.addCleanup(evolution.store.close)
        self.assertIsNone(evolution.rollback('harness'))
        policy,provenance=load_active_policy(self.campaign,None)
        self.assertEqual(policy,{});self.assertEqual(provenance['status'],'builtin_after_rollback')
        self.assertEqual(read_json(self.campaign/'versions/candidate/policy.json'),{'retrieval_limit':25})

    def test_validator_receives_frozen_cost_prefix_including_own_reservation(self):
        path=self.config();config=read_json(path)
        for stage,rule in config['plan']['stages'].items():
            script=Path(rule['command'][1])
            script.write_text(script.read_text()+"\nimport json\ncost=json.loads(Path(sys.argv[2]).read_text())\nassert cost['pending']==1\nassert cost['charged_or_reserved']>=2\n")
            rule['command'].append('{cost_context}')
        write_json(path,config)
        result=autoresearch(self.campaign,path,agent=self.agent)
        self.assertEqual(result['status'],'promoted')
        from rec_harness.cost_audit import verify_cost_prefix
        store=Store(self.campaign/'memory');self.addCleanup(store.close)
        for stage,expected in [('prefix_audit',2),('development',3),('confirmation',4)]:
            prefix=read_json(self.campaign/'candidates/candidate/validation'/stage/'controller-cost.json')
            self.assertEqual(verify_cost_prefix(store,prefix),expected)

    def test_changed_validator_dependency_is_rejected_before_execution_or_charge(self):
        config=read_json(self.config());dependency=self.root/'validator-helper.py';dependency.write_text('version=1\n')
        config['plan']['stages']['prefix_audit']['validator_files']=[str(dependency)]
        evolution=Evolution(self.campaign);self.addCleanup(evolution.store.close)
        evolution.propose('candidate','harness',config['context'],config['plan'],agent=self.agent)
        before=evolution.store.budget();dependency.write_text('version=2\n')
        with self.assertRaisesRegex(ContractError,'frozen validator changed'):
            evolution.validate('candidate','prefix_audit')
        self.assertEqual(evolution.store.budget(),before)

    def test_role_scoped_prompt_candidate_passes_generation_contract(self):
        class ScopedOptimizer(Optimizer):
            def run(self,*args):
                result=super().run(*args)
                result['editable_scope']=['prompts.lead']
                result['patch_diff']=json.dumps({'prompts':{'lead':'Compare recorded source provenance.'}})
                return result
        agent=ScopedOptimizer()
        result=autoresearch(self.campaign,self.config(),agent=agent)
        self.assertEqual(result['status'],'promoted')
        self.assertEqual(read_json(self.campaign/'versions/candidate/policy.json'),
                         {'prompts':{'lead':'Compare recorded source provenance.'}})
