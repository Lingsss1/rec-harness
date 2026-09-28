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
    def test_harness_rationale_is_archived_and_linked_to_frozen_trace(self):
        path=self.config(reject='development');config=read_json(path)
        config['context']['decision_traces']=[{'trace_id':'trace','harness_version':'source-v1','visible_state':{'old':'input'}}]
        write_json(path,config)
        result=autoresearch(self.campaign,path,agent=self.agent)
        self.assertEqual(result['status'],'rejected')
        evolution=Evolution(self.campaign)
        try:
            snapshot=evolution.store.snapshot()
            hypothesis=next(n for n in snapshot['nodes'] if n['type']=='DecisionFailureHypothesis')
            self.assertEqual(hypothesis['attrs']['origin'],'agent_hypothesis')
            self.assertEqual(hypothesis['attrs']['status'],'provisional')
            trace=next(n for n in snapshot['nodes'] if n['attrs'].get('external_trace_id')=='trace')
            self.assertNotIn('visible_state',trace['attrs'])
            generated=next(e for e in snapshot['edges'] if e['relation']=='GENERATED_BY')
            self.assertEqual(generated['source'],trace['id'])
            harness=next(n for n in snapshot['nodes'] if n['id']==generated['target'])
            self.assertEqual(harness['attrs']['external_version'],'source-v1')
            self.assertEqual(harness['attrs']['lifecycle_event'],'observed_in_trace')
            support=next(e for e in snapshot['edges'] if e['relation']=='SUPPORTED_BY')
            self.assertEqual(support['target'],trace['id'])
            self.assertEqual(support['attrs']['evidence_refs'],[trace['id']])
            self.assertTrue(any(e['relation']=='ADDRESSES' and e['target']==hypothesis['id'] for e in snapshot['edges']))
            proposal=evolution.artifacts.verify(hypothesis['attrs']['proposal_ref'])
            self.assertEqual(read_json(proposal)['trace_refs'],['trace'])
            evolution.verify()
            proposal.write_text('{}')
            with self.assertRaisesRegex(ContractError,'corrupt artifact'):evolution.verify()
        finally:evolution.store.close()

    def test_invalid_plan_stops_without_inventing_candidate_graph(self):
        evolution=Evolution(self.campaign)
        try:
            evolution.store.configure_budget(12)
            result=evolution.run('invalid','tool',{}, {},agent=self.agent)
            self.assertEqual(result['status'],'rejected')
            self.assertEqual(self.agent.calls,0)
            self.assertEqual(evolution.store.snapshot()['nodes'],[])
            evolution.verify()
        finally:evolution.store.close()

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

    def test_committed_generation_recovers_before_candidate_projection_without_new_call(self):
        from rec_harness import evolution as module
        original=module.write_json
        def interrupt(path,value):
            if Path(path).name=='candidate.json' and value.get('status')=='implemented':raise KeyboardInterrupt()
            return original(path,value)
        with patch.object(module,'write_json',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(),agent=self.agent)
        self.assertEqual(self.agent.calls,1)
        marker=self.campaign/'candidates/candidate/candidate.json'
        self.assertEqual(read_json(marker)['status'],'proposed')
        store=Store(self.campaign/'memory')
        try:
            self.assertEqual(sum(e['kind']=='candidate_implemented' for e in store.events()),1)
            self.assertEqual(store.budget()['charged_or_reserved'],1)
        finally:store.close()
        # The event also supports reconstructing a missing projection.
        marker.unlink()
        result=autoresearch(self.campaign,agent=self.agent)
        self.assertEqual(result['status'],'promoted')
        self.assertEqual(result['budget']['charged_or_reserved'],4)
        self.assertEqual(self.agent.calls,1)
        self.assertEqual(autoresearch(self.campaign,agent=self.agent),result)
        self.assertEqual(self.agent.calls,1)

    def test_committed_generation_survives_unsettled_cost_without_new_generation(self):
        original=Store.settle
        interrupted=[]
        def interrupt(store,task,actual,status):
            if '-generation-' in task and not interrupted:
                interrupted.append(True);raise KeyboardInterrupt()
            return original(store,task,actual,status)
        with patch.object(Store,'settle',interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(),agent=self.agent)
        self.assertEqual(self.agent.calls,1)
        result=autoresearch(self.campaign,agent=self.agent)
        self.assertEqual(result['status'],'promoted')
        self.assertEqual(result['budget']['charged_or_reserved'],4)
        self.assertEqual(self.agent.calls,1)
        store=Store(self.campaign/'memory')
        try:
            row=store.db.execute("SELECT status,actual FROM budget WHERE id='candidate-generation-1'").fetchone()
            self.assertEqual(row['status'],'interrupted');self.assertIsNone(row['actual'])
            self.assertEqual(sum(e['kind']=='candidate_implemented' for e in store.events()),1)
        finally:store.close()

    def test_successful_raw_generation_receipts_resume_without_a_second_call(self):
        from rec_harness.agents import schema_for
        class Completed(Optimizer):
            def run(self,role,context,directory):
                response=super().run(role,context,directory)
                for name,value in [('context.json',context),('request.json',{'role':role,'context':context}),
                                   ('schema.json',schema_for(role)),('response.json',response),('resources.json',{'status':'success'})]:
                    write_json(Path(directory)/name,value)
                raise KeyboardInterrupt()
        agent=Completed()
        with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(),agent=agent)
        result=autoresearch(self.campaign,agent=agent)
        self.assertEqual(result['status'],'promoted');self.assertEqual(agent.calls,1)
        self.assertEqual(result['budget']['charged_or_reserved'],4)
        self.assertEqual(autoresearch(self.campaign,agent=agent),result)
        evolution=Evolution(self.campaign)
        try:
            events=evolution.store.events()
            receipts=[e['payload']['receipts'] for e in events if e['kind']=='candidate_generation_recovered']
            self.assertEqual(len(receipts),1)
            for ref in receipts[0].values():evolution.artifacts.verify(ref)
        finally:evolution.store.close()

    def test_tool_receipt_recovery_binds_source_across_another_interruption(self):
        from rec_harness.agents import schema_for
        config=read_json(self.config());plan=config['plan']
        rule=next(iter(plan['stages'].values()))
        plan['stages']={stage:dict(rule) for stage in STAGES['tool']}
        class Tool:
            calls=0
            def run(self,role,context,directory):
                self.calls+=1;model=Path(directory)/'model';model.mkdir(parents=True)
                (model/'diagnostic.py').write_text('def measure(request): return {}\n')
                response={'changed_files':['diagnostic.py'],'implementation_checks':[],'limitations':['fixture']}
                for name,value in [('context.json',context),('request.json',{'role':role,'context':context}),
                    ('schema.json',schema_for(role)),('response.json',response),('resources.json',{'status':'success'})]:
                    write_json(Path(directory)/name,value)
                raise KeyboardInterrupt()
        agent=Tool();evolution=Evolution(self.campaign);self.addCleanup(evolution.store.close)
        with self.assertRaises(KeyboardInterrupt):evolution.propose('candidate','tool',{},plan,agent=agent)
        original=evolution.store.event
        def interrupt(kind,payload):
            result=original(kind,payload)
            if kind=='candidate_generation_recovered':raise KeyboardInterrupt()
            return result
        with patch.object(evolution.store,'event',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):evolution.propose('candidate','tool',{},plan,agent=agent)
        events=evolution.store.events()
        recovered=next(e['payload'] for e in events if e['kind']=='candidate_generation_recovered')
        self.assertIn('model/diagnostic.py',recovered['receipts'])
        source=self.campaign/'candidates/candidate/development/model/diagnostic.py'
        source.write_text('def measure(request): return {"changed": 1}\n')
        with self.assertRaisesRegex(ContractError,'recovered generation artifacts changed'):
            evolution.propose('candidate','tool',{},plan,agent=agent)
        self.assertEqual(agent.calls,1)
        self.assertEqual(evolution.store.budget()['charged_or_reserved'],1)

    def test_unknown_generation_outcome_stops_instead_of_resubmitting(self):
        class Interrupted(Optimizer):
            def run(self,role,context,directory):
                super().run(role,context,directory)
                raise KeyboardInterrupt()
        agent=Interrupted()
        with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(),agent=agent)
        result=autoresearch(self.campaign,agent=agent)
        self.assertEqual(result['status'],'rejected')
        self.assertIn('not automatic resubmission',result['reason'])
        self.assertEqual(result['budget']['charged_or_reserved'],1)
        self.assertEqual(agent.calls,1)
        self.assertEqual(autoresearch(self.campaign,agent=agent),result)
        self.assertEqual(agent.calls,1)

    def test_graph_projection_recovers_after_terminal_event_without_new_generation(self):
        from rec_harness import evolution_graph
        with patch.object(evolution_graph,'sync_lifecycle',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(),agent=self.agent)
        count=self.agent.calls
        result=autoresearch(self.campaign,agent=self.agent)
        self.assertEqual(result['status'],'promoted');self.assertEqual(self.agent.calls,count)
        evolution=Evolution(self.campaign)
        try:
            self.assertTrue(evolution.store.snapshot()['nodes']);evolution.verify()
            before=evolution.store.verify()
            evolution_graph.sync_lifecycle(evolution.store)
            self.assertEqual(evolution.store.verify(),before)
        finally:evolution.store.close()

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
                contract=context['context']['diagnostic_contract']
                assert contract['entrypoint']=='diagnostic.py:measure(request)'
                assert set(contract['response'])=={'measurements','limitations'}
                assert 'runs' in contract['request']
                target=Path(directory)/'model';target.mkdir(parents=True)
                (target/'diagnostic.py').write_text('def measure(request):\n    values = [r["metrics"]["quality"] for r in request["runs"]]\n    return {"measurements": {"spread": max(values) - min(values)}, "limitations": ["fixture: aggregate range only"]}\n')
                return {'changed_files':['diagnostic.py']}
        agent=ToolAgent();result=autoresearch(self.campaign,path,agent=agent)
        self.assertEqual(result['status'],'promoted');self.assertEqual(result['budget']['charged_or_reserved'],7)
        store=Store(self.campaign/'memory')
        try:
            graph=store.snapshot();types=[n['type'] for n in graph['nodes']]
            self.assertEqual(types.count('ToolCandidate'),1);self.assertEqual(types.count('ToolValidation'),6)
            self.assertEqual(types.count('ToolVersion'),1);self.assertNotIn('Claim',types)
            self.assertTrue(all(n['attrs']['independent_unit'] is False for n in graph['nodes']))
            self.assertTrue(any(e['relation']=='PROMOTED_FROM' for e in graph['edges']))
        finally:store.close()
        self.assertEqual(autoresearch(self.campaign,agent=agent),result);self.assertEqual(agent.calls,1)
        from rec_harness.diagnostics import freeze_tool
        from rec_harness.core import Artifacts
        artifacts=Artifacts(self.root/'new-campaign/objects')
        frozen=freeze_tool({'registry':self.campaign,'identity':config['plan']['identity'],
            'execution':{'recipe':{},'python':'python3','uid_registry':'unused'},'cost':1,'timeout':5},artifacts)
        self.assertEqual(frozen['candidate_id'],'candidate')
        for ref in frozen['implementation'].values():artifacts.verify(ref)

    def test_new_tool_cannot_override_runtime_authoring_contract(self):
        from rec_harness.evolution_auto import AutonomousEvolution
        config=read_json(self.config());config['kind']='tool'
        config['plan']['tool_contract']='public-aggregate-measurement-v1'
        config['context']['diagnostic_contract']={'entrypoint':'wrong.py:run'}
        with self.assertRaisesRegex(ContractError,'authoring contract differs'):
            AutonomousEvolution.init(self.campaign,config)
        self.assertFalse((self.campaign/'auto-lock.json').exists())
        self.assertEqual(self.agent.calls,0)

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
        self.assertTrue(any(n['attrs'].get('kind')=='evolution_rollback' for n in evolution.store.snapshot()['nodes']))
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
