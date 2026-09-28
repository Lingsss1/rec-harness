"""Project orchestration fixtures; constant validator reports are not online evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.autonomous import autoresearch
from rec_harness.core import Artifacts,ContractError,read_json,write_json
from rec_harness.evolution import STAGES
from rec_harness.project import ResearchProject
from rec_harness.store import Store
from test_runtime import ROOT,ScriptedAgent


class ProjectAgent(ScriptedAgent):
    def run(self,role,context,directory):
        if role=='tool':
            model=Path(directory)/'model';model.mkdir(parents=True,exist_ok=True)
            (model/'diagnostic.py').write_text('def measure(request): return {}\n')
            return {'changed_files':['diagnostic.py'],'implementation_checks':[],'limitations':['fixture']}
        if role in {'optimizer','optimizer_router'}:
            trace=context['context']['decision_traces'][0]['trace_id']
            response={'patch_id':'patch','failure_pattern':'fixture','trace_refs':[trace],
                'causal_hypothesis':'fixture','editable_scope':['retrieval_limit'],'patch_diff':'{"retrieval_limit":25}',
                'expected_decision_change':'fixture','regression_risks':['fixture'],'validation_plan':'frozen','rollback_version':'base'}
            return {'kind':'harness_patch','payload_json':json.dumps(response)} if role=='optimizer_router' else response
        return super().run(role,context,directory)

class DiscoverAgent(ProjectAgent):
    def __init__(self):self.measurement_calls=[]
    def run(self,role,context,directory):
        if role!='measurement':return super().run(role,context,directory)
        self.measurement_calls.append(context)
        measured=context.get('diagnostic') is not None
        return {'assessment':'sufficient' if measured else 'measurement_gap',
            'rationale':'Fixture: quantify variation before choosing a model change',
            'evidence_refs':context['allowed_evidence_refs'],
            'prerequisites':{k:{'status':'satisfied','rationale':'fixture evidence'} for k in ['invocation','implementation','data','applicability']},
            'requirements':[] if measured else [{'measurement':'spread','blocked_decision':'choose model change',
                'competing_explanations':['capacity','sample noise'],'cheapest_test':'aggregate range',
                'development_budget':10,'acceptance_contract':'independent reference and decision utility'}]}


class ProjectTests(unittest.TestCase):
    def test_tool_rejects_undeclared_changes_or_deletion_of_inherited_files(self):
        source=self.root/'tool-source';source.mkdir()
        (source/'diagnostic.py').write_text('def measure(request): return {}\n')
        (source/'helper.py').write_text('SCALE = 0.5\n')
        config=read_json(self.config('tool'))
        config['tool'].update(source=str(source),model_files=['diagnostic.py','helper.py'])
        path=self.root/'invalid-tool.json';write_json(path,config)
        for deletion in [False,True]:
            with self.subTest(deletion=deletion):
                class UndeclaredAgent(ProjectAgent):
                    def run(agent,role,context,directory):
                        response=super().run(role,context,directory)
                        if role=='tool':
                            helper=Path(directory)/'model/helper.py'
                            if deletion:helper.unlink()
                            else:helper.write_text('SCALE = 99\n')
                        return response
                campaign=self.root/('invalid-'+str(deletion))
                result=autoresearch(campaign,path,agent=UndeclaredAgent())
                self.assertEqual(result['state']['stop_reason'],'tool_rejected')
                self.assertFalse((campaign/'episodes/tool-1/campaign/active-tool.json').exists())

    def test_tool_then_harness_preserve_all_three_states_across_interruption(self):
        config=read_json(self.config('tool'))
        config['harness']=read_json(self.config('harness'))['harness']
        config['max_model_episodes']=3
        path=self.root/'combined.json';write_json(path,config)
        calls=[]
        class RecordingAgent(ProjectAgent):
            def run(agent,role,context,directory):
                calls.append(role)
                return super().run(role,context,directory)
        agent=RecordingAgent();transition=ResearchProject.transition
        def interrupt_after_harness(project,state,episode=None):
            transition(project,state,episode)
            if episode and episode['kind']=='harness':raise KeyboardInterrupt()
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.cpu):
            with patch.object(ResearchProject,'transition',interrupt_after_harness):
                with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,path,agent=agent)
            before=list(calls)
            report=autoresearch(self.campaign,agent=agent)
            self.assertEqual([e['kind'] for e in report['episodes']],['model','tool','model','harness','model'])
            self.assertNotIn('tool',calls[len(before):]);self.assertNotIn('optimizer',calls[len(before):])
            count=len(calls)
            self.assertEqual(autoresearch(self.campaign,agent=agent),report)
            self.assertEqual(len(calls),count)
        final=self.campaign/'episodes/model-3/campaign'
        prior=self.campaign/'episodes/model-2/campaign'
        final_config=read_json(final/'campaign.json')
        self.assertEqual(final_config['research_policy'],{'retrieval_limit':25})
        self.assertEqual(final_config['diagnostic_tool']['candidate_id'],'project-tool-1')
        self.assertEqual(final_config['measurement_requirements'],read_json(prior/'campaign.json')['measurement_requirements'])
        self.assertEqual(read_json(final/'research_archive/manifest.json')['root_workspace'],read_json(prior/'state.json')['workspace'])
        store=Store(final/'memory')
        try:
            coverage=[e['payload'] for e in store.events() if e['kind']=='measurement_output_observed']
            self.assertEqual(len(coverage),1)
            self.assertEqual(coverage[0]['tool_version'],'project-tool-1')
            self.assertEqual(coverage[0]['source_measurement_gap_ids'],[report['state']['measurement_obligations'][0]['measurement_gap_id']])
        finally:store.close()
        project=ResearchProject(self.campaign,agent=agent)
        try:self.assertEqual(len(project.verify()['verified_episodes']),5)
        finally:project.store.close()

    def test_successive_harness_patches_preserve_current_policy_and_other_prompts(self):
        path=self.config('harness');config=read_json(path)
        config.update(max_model_episodes=3,max_harness_updates=2)
        config['model']['research_policy']={'prompts':{'evidence':'Existing evidence guidance'}}
        write_json(path,config);bases=[]
        class IncrementalOptimizer(ProjectAgent):
            def run(agent,role,context,directory):
                response=super().run('optimizer' if role=='optimizer_router' else role,context,directory)
                if role in {'optimizer','optimizer_router'}:
                    bases.append(context['context']['base_policy'])
                    if len(bases)==2:
                        response.update(editable_scope=['prompts.lead'],patch_diff=json.dumps({'prompts':{'lead':'Check contrary evidence'}}))
                return {'kind':'harness_patch','payload_json':json.dumps(response)} if role=='optimizer_router' else response
        agent=IncrementalOptimizer()
        report=autoresearch(self.campaign,path,agent=agent)
        self.assertEqual([e['kind'] for e in report['episodes']],['model','harness','model','harness','model'])
        expected={'prompts':{'evidence':'Existing evidence guidance'},'retrieval_limit':25}
        self.assertEqual(bases,[{'prompts':{'evidence':'Existing evidence guidance'}},expected])
        final=read_json(self.campaign/'episodes/model-3/campaign/campaign.json')['research_policy']
        self.assertEqual(final,{'retrieval_limit':25,'prompts':{'evidence':'Existing evidence guidance','lead':'Check contrary evidence'}})
        second=self.campaign/'episodes/harness-2/campaign'
        candidate=read_json(second/'candidates/project-harness-2/candidate.json')
        from rec_harness.core import Artifacts
        artifacts=Artifacts(second/'objects')
        self.assertEqual(read_json(artifacts.verify(candidate['implementation']['policy.json'])),final)
        proposal=read_json(artifacts.verify(candidate['proposal_ref']))
        self.assertEqual(json.loads(proposal['patch_diff']),{'prompts':{'lead':'Check contrary evidence'}})
        self.assertEqual(autoresearch(self.campaign,agent=agent),report);self.assertEqual(len(bases),2)
        project=ResearchProject(self.campaign,agent=agent)
        try:self.assertEqual(len(project.verify()['verified_episodes']),5)
        finally:project.store.close()

    def test_two_tool_updates_inherit_validated_source_and_remeasure_all_obligations(self):
        path=self.config('tool');config=read_json(path)
        config.update(max_model_episodes=3,max_tool_updates=2)
        first=config['model']['measurement_requirements'][0]
        config['model']['measurement_requirements'].append({**first,'measurement':'tail'})
        write_json(path,config)
        inherited=[]
        class ExtendingAgent(ProjectAgent):
            def run(agent,role,context,directory):
                if role=='tool':
                    source=Path(directory)/'model/diagnostic.py'
                    inherited.append(source.read_text() if source.exists() else None)
                    response=super().run(role,context,directory)
                    source.write_text(source.read_text()+'# generation '+str(len(inherited))+'\n')
                    helper=source.parent/'helper.py'
                    if len(inherited)==1:
                        helper.write_text('SCALE = 0.5\n');response['changed_files'].append('helper.py')
                    else:self.assertEqual(helper.read_text(),'SCALE = 0.5\n')
                    return response
                return super().run(role,context,directory)
        def measure(directory,*args,**kwargs):
            values={'spread':0.5}
            if 'model-3' in Path(directory).parts:values['tail']=0.25
            output=Path(directory)/'results/measurement.json'
            write_json(output,{'measurements':values,'limitations':['fixture']})
            return {'process':{'status':'success'},'collection':{'artifacts':{'measurement.json':kwargs['artifacts'].put(output)}}}
        agent=ExtendingAgent()
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=measure):
            report=autoresearch(self.campaign,path,agent=agent)
            self.assertEqual([e['kind'] for e in report['episodes']],['model','tool','model','tool','model'])
            self.assertEqual(autoresearch(self.campaign,agent=agent),report)
        self.assertEqual(len(inherited),2);self.assertIsNone(inherited[0])
        self.assertIn('# generation 1',inherited[1])
        first_tool=read_json(self.campaign/'episodes/tool-1/campaign/candidates/project-tool-1/candidate.json')
        second_tool=read_json(self.campaign/'episodes/tool-2/campaign/evolution-auto.json')
        self.assertEqual(second_tool['context']['tool_predecessor']['implementation_hash'],first_tool['implementation_hash'])
        self.assertEqual(second_tool['source_snapshot'],first_tool['implementation'])
        published=read_json(self.campaign/'episodes/tool-2/campaign/candidates/project-tool-2/candidate.json')
        self.assertEqual(published['implementation']['helper.py'],first_tool['implementation']['helper.py'])
        self.assertEqual((self.campaign/'episodes/tool-2/campaign/versions/project-tool-2/helper.py').read_text(),'SCALE = 0.5\n')
        store=Store(self.campaign/'episodes/tool-2/campaign/memory')
        try:
            graph=store.snapshot();edge=next(e for e in graph['edges'] if e['relation']=='BASED_ON')
            inherited_version=next(n for n in graph['nodes'] if n['id']==edge['target'])
            self.assertEqual(inherited_version['attrs']['implementation_hash'],first_tool['implementation_hash'])
            self.assertEqual(inherited_version['attrs']['lifecycle_event'],'inherited_source')
        finally:store.close()
        final=self.campaign/'episodes/model-3/campaign'
        self.assertEqual({r['measurement'] for r in read_json(final/'campaign.json')['measurement_requirements']},{'spread','tail'})
        tool=read_json(final/'campaign.json')['diagnostic_tool'];registration=tool['registration']
        self.assertEqual(set(registration['validation_stages']),set(STAGES['tool']))
        self.assertTrue(all(r['status']=='passed' and len(r['report_hash'])==64 for r in registration['validation_stages'].values()))
        self.assertEqual(registration['source_tool_predecessor']['implementation_hash'],first_tool['implementation_hash'])
        self.assertIsNone(registration['rollback_candidate_id'])
        self.assertEqual(registration['declared_call_cost'],config['tool_runtime']['cost'])
        store=Store(final/'memory')
        try:
            observed=[e['payload'] for e in store.events() if e['kind']=='measurement_output_observed']
            self.assertEqual({r['measurement'] for r in observed},{'spread','tail'})
            self.assertTrue(all(r['tool_version']=='project-tool-2' for r in observed))
            diagnostic=next(e['payload'] for e in store.events() if e['kind']=='diagnostic_completed')
            self.assertEqual(diagnostic['tool_registration'],registration)
            version=next(n for n in store.snapshot()['nodes'] if n['type']=='ToolVersion' and n['attrs'].get('version')=='project-tool-2')
            self.assertEqual(version['attrs']['registration'],registration)
        finally:store.close()
        project=ResearchProject(self.campaign,agent=agent)
        try:self.assertEqual(len(project.verify()['verified_episodes']),5)
        finally:project.store.close()
        from rec_harness.evolution_auto import AutonomousEvolution
        bad=dict(second_tool);bad.pop('source_snapshot')
        changed=self.root/'changed-source';changed.mkdir()
        (changed/'diagnostic.py').write_text('def measure(request): return {}\n')
        bad.update(source=str(changed),model_files=['diagnostic.py'])
        with self.assertRaisesRegex(ContractError,'predecessor differs'):
            AutonomousEvolution.init(self.root/'bad-inheritance',bad)
        self.assertFalse((self.root/'bad-inheritance/auto-lock.json').exists())

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.campaign=self.root/'project';self.agent=ProjectAgent()

    def config(self,kind):
        model=read_json(ROOT/'examples/toy/campaign.json');model.update(source=str(ROOT/'examples/toy'),max_rounds=1)
        identity={k:'fixture' for k in ['base_llm','tools','start','graph','budget','protocol']};stages={}
        for stage in STAGES[kind]:
            report={'passed':True,'gain':1,'regression':0,'total_cost':7,'evaluation_ids':[stage+'-a',stage+'-b'],
                'identity':identity,'comparison_groups':['no_tool','existing_tool','candidate_tool'] if kind=='tool' else ['baseline','candidate'],'online':True}
            script=self.root/(stage+'.py');script.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('+repr(json.dumps(report))+')\n')
            stages[stage]={'command':['{python}',str(script),'{output}'],'timeout':5,'cost':1,'minimum_gain':0.1,'maximum_regression':0,'maximum_total_cost':20}
        evolution={'adapter':'research-evolution-v1','kind':kind,'candidate_id':'placeholder',
            'plan':{'budget':20,'agent_cost':1,'development_ids':['source'],'identity':identity,'stages':stages,'rollback_version':'base'}}
        if kind=='tool':
            evolution['plan']['tool_contract']='public-aggregate-measurement-v1'
            model['measurement_requirements']=[{'measurement':'spread','blocked_decision':'choose model change',
                'competing_explanations':['capacity','sample noise'],'cheapest_test':'probe','development_budget':20,'acceptance_contract':'independent decision utility'}]
        config={'adapter':'research-project-v1','task_id':'project','max_model_episodes':2,'model':model,kind:evolution}
        if kind=='tool':config['tool_runtime']={'execution':{'recipe':{},'python':'unused','uid_registry':'unused'},'cost':1,'timeout':5}
        path=self.root/'config.json';write_json(path,config);return path

    def cpu(self,directory,*args,**kwargs):
        path=Path(directory)/'results/measurement.json';write_json(path,{'measurements':{'spread':0.5},'limitations':['fixture']})
        return {'process':{'status':'success'},'collection':{'artifacts':{'measurement.json':kwargs['artifacts'].put(path)}}}

    def test_transport_stopped_child_cannot_trigger_evolution_or_finish_project(self):
        config=read_json(self.config('harness'));ResearchProject.init(self.campaign,config)
        project=ResearchProject(self.campaign,agent=self.agent)
        try:
            result={'state':{'status':'stopped','stop_reason':'role_transport_failure','round':1}}
            with patch.object(project,'child',return_value=result) as child:
                for _ in range(2):
                    with self.assertRaisesRegex(ContractError,'explicitly reopen the child'):project.run()
                self.assertEqual(child.call_count,2)
                self.assertTrue(all(call.args[0].name=='model-1' for call in child.call_args_list))
            state=read_json(self.campaign/'state.json')
            self.assertEqual(state['phase'],'model');self.assertEqual(state['status'],'ready')
            events=project.store.events()
            self.assertEqual(sum(e['kind']=='project_child_transport_stopped' for e in events),1)
            self.assertFalse(any(e['kind']=='project_transition' and 'episode' in e['payload'] for e in events))
            # Repair can return through the same model phase; stop this fixture
            # just before it would invoke the external optimizer.
            original=project.transition
            def stop_after_model(state,episode=None):
                original(state,episode)
                raise KeyboardInterrupt()
            with patch.object(project,'child',return_value={'state':{'stop_reason':'max_rounds'}}),patch.object(project,'transition',side_effect=stop_after_model):
                with self.assertRaises(KeyboardInterrupt):project.run()
            self.assertEqual(read_json(self.campaign/'state.json')['phase'],'harness')
        finally:project.store.close()

    def test_model_gap_tool_validation_new_episode_and_remeasurement(self):
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.cpu):
            report=autoresearch(self.campaign,self.config('tool'),agent=self.agent)
            self.assertEqual([r['kind'] for r in report['episodes']],['model','tool','model'])
            self.assertEqual(report['state']['tool_updates'],1)
            tool_store=Store(self.campaign/'episodes/tool-1/campaign/memory')
            try:
                tool_graph=tool_store.snapshot()
                self.assertTrue(any(n['type']=='MeasurementGap' for n in tool_graph['nodes']))
                self.assertTrue(any(e['relation']=='ADDRESSES' for e in tool_graph['edges']))
            finally:tool_store.close()
            first=read_json(self.campaign/'episodes/model-1/campaign/run-report.json')
            second=read_json(self.campaign/'episodes/model-2/campaign/run-report.json')
            self.assertEqual(first['state']['stop_reason'],'measurement_required')
            self.assertEqual(second['state']['stop_reason'],'max_rounds')
            self.assertEqual(second['state']['incumbent'],'attempt-1')
            self.assertEqual(autoresearch(self.campaign,agent=self.agent),report)
            project=ResearchProject(self.campaign,agent=self.agent)
            try:self.assertEqual(len(project.verify()['verified_episodes']),3)
            finally:project.store.close()

    def test_harness_successor_preserves_incumbent_and_transition_recovers(self):
        original=ResearchProject.transition;once=[]
        def interrupt(instance,state,episode=None):
            original(instance,state,episode)
            if episode and episode['kind']=='harness' and not once:
                once.append(True);raise KeyboardInterrupt()
        with patch.object(ResearchProject,'transition',interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config('harness'),agent=self.agent)
        report=autoresearch(self.campaign,agent=self.agent)
        self.assertEqual([r['kind'] for r in report['episodes']],['model','harness','model'])
        parent=read_json(self.campaign/'episodes/model-1/campaign/state.json')
        child=self.campaign/'episodes/model-2/campaign'
        self.assertEqual(read_json(child/'research_archive/manifest.json')['root_workspace'],parent['workspace'])
        self.assertEqual(read_json(child/'campaign.json')['research_policy'],{'retrieval_limit':25})
        self.assertTrue(read_json(child/'campaign.json')['initial_memory'])
        self.assertEqual(autoresearch(self.campaign,agent=self.agent),report)

    def test_rejected_tool_never_activates_or_starts_a_successor(self):
        path=self.config('tool');config=read_json(path)
        config['tool']['plan']['stages']['utility']['minimum_gain']=2
        write_json(path,config)
        report=autoresearch(self.campaign,path,agent=self.agent)
        self.assertEqual(report['state']['stop_reason'],'tool_rejected')
        self.assertEqual([r['kind'] for r in report['episodes']],['model','tool'])
        self.assertNotIn('active_tool',report['state'])
        self.assertEqual(autoresearch(self.campaign,agent=self.agent),report)

    def test_completed_child_before_project_commit_is_reused(self):
        original=ResearchProject.transition;once=[];calls=[]
        class Count(ProjectAgent):
            def run(self,role,context,directory):
                calls.append(role);return super().run(role,context,directory)
        agent=Count()
        def interrupt(instance,state,episode=None):
            if episode and not once:
                once.append(True);raise KeyboardInterrupt()
            original(instance,state,episode)
        with patch.object(ResearchProject,'transition',interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config('harness'),agent=agent)
        self.assertEqual(len(calls),6)
        report=autoresearch(self.campaign,agent=agent)
        self.assertEqual(len(calls),13);self.assertEqual(len(report['episodes']),3)

    def test_discovered_gap_without_preregistered_requirements_routes_tool_and_remeasures(self):
        path=self.config('tool');config=read_json(path)
        del config['model']['measurement_requirements'];write_json(path,config)
        agent=DiscoverAgent();calls=agent.measurement_calls
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.cpu):
            report=autoresearch(self.campaign,path,agent=agent)
            self.assertEqual([r['kind'] for r in report['episodes']],['model','tool','model'])
            self.assertEqual(len(calls),2)
            self.assertIn('spread',calls[-1]['diagnostic']['output']['measurements'])
            self.assertEqual(autoresearch(self.campaign,agent=agent),report)
            self.assertEqual(len(calls),2)
        context=read_json(self.campaign/'episodes/tool-1/campaign/candidates/project-tool-1/context.json')
        self.assertEqual(context['measurement_review']['assessment'],'measurement_gap')
        child=self.campaign/'episodes/model-2/campaign'
        rule=read_json(child/'campaign.json')['measurement_requirements'][0]
        gap_id=context['measurement_gap']['measurement_gap_id']
        self.assertEqual(rule['source_measurement_gap_ids'],[gap_id])
        store=Store(child/'memory')
        try:
            records=[e['payload'] for e in store.events() if e['kind']=='measurement_output_observed']
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]['source_measurement_gap_ids'],[gap_id])
            self.assertEqual(records[0]['value'],0.5)
            self.assertFalse(records[0]['independent_unit'])
            graph=store.snapshot();known={n['id']:n for n in graph['nodes']}
            source=known[records[0]['evidence_dependency_ids'][0]]
            self.assertEqual(source['attrs']['output']['measurements']['spread'],0.5)
            self.assertEqual(source['attrs']['implementation_hash'],records[0]['implementation_hash'])
            from types import SimpleNamespace
            from rec_harness.measurement_gaps import record_coverage
            config=read_json(child/'campaign.json')
            research=SimpleNamespace(store=store,task=config['task_id'],protocol_hash=config['protocol_hash'])
            state={'incumbent':source['attrs']['model_version']}
            before=store.verify()
            record_coverage(research,state,source['attrs'],rule)
            self.assertEqual(store.verify(),before)
            forged=json.loads(json.dumps(source['attrs']))
            forged['output']['measurements']['spread']=100
            with self.assertRaisesRegex(ContractError,'matching diagnostic evidence'):
                record_coverage(research,state,forged,rule)
        finally:store.close()

    def test_promoted_tool_missing_discovered_output_blocks_successor(self):
        path=self.config('tool');config=read_json(path)
        del config['model']['measurement_requirements'];write_json(path,config)
        agent=DiscoverAgent()
        def wrong_measurement(directory,*args,**kwargs):
            path=Path(directory)/'results/measurement.json'
            write_json(path,{'measurements':{'unrelated':0.5},'limitations':['fixture']})
            return {'process':{'status':'success'},'collection':{'artifacts':{'measurement.json':kwargs['artifacts'].put(path)}}}
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=wrong_measurement):
            report=autoresearch(self.campaign,path,agent=agent)
            self.assertEqual(report['state']['stop_reason'],'measurement_unresolved')
            self.assertEqual(len(agent.measurement_calls),1)
            child=self.campaign/'episodes/model-2/campaign'
            self.assertEqual(read_json(child/'campaign.json')['measurement_requirements'][0]['measurement'],'spread')
            self.assertEqual(read_json(child/'run-report.json')['state']['stop_reason'],'measurement_required')
            store=Store(child/'memory')
            try:self.assertFalse(any(e['kind']=='measurement_output_observed' for e in store.events()))
            finally:store.close()
            decision_context=read_json(next((child/'research_archive/decisions').glob('*/context.json')))
            self.assertEqual(decision_context['diagnostic']['status'],'success')
            self.assertEqual(set(decision_context['diagnostic']['output']['measurements']),{'unrelated'})
            self.assertEqual(decision_context['measurement_gaps'][0]['required_information'],'spread')
            self.assertEqual(decision_context['measurement_gaps'][0]['related_gap_ids'],
                read_json(child/'campaign.json')['measurement_requirements'][0]['source_measurement_gap_ids'])
            self.assertFalse(any('hypothesis' in p.name for p in (child/'tasks').iterdir()))
            self.assertEqual(autoresearch(self.campaign,agent=agent),report)
        self.assertEqual(report['state']['measurement_obligations'][0]['required_information'],'spread')
        project=ResearchProject(self.campaign,agent=agent)
        try:self.assertEqual(len(project.verify()['verified_episodes']),3)
        finally:project.store.close()
