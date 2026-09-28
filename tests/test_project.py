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
from test_runtime import ROOT,ScriptedAgent


class ProjectAgent(ScriptedAgent):
    def run(self,role,context,directory):
        if role=='tool':
            model=Path(directory)/'model';model.mkdir(parents=True,exist_ok=True)
            (model/'diagnostic.py').write_text('def measure(request): return {}\n')
            return {'changed_files':['diagnostic.py'],'implementation_checks':[],'limitations':['fixture']}
        if role=='optimizer':
            trace=context['context']['decision_traces'][0]['trace_id']
            return {'patch_id':'patch','failure_pattern':'fixture','trace_refs':[trace],
                'causal_hypothesis':'fixture','editable_scope':['retrieval_limit'],'patch_diff':'{"retrieval_limit":25}',
                'expected_decision_change':'fixture','regression_risks':['fixture'],'validation_plan':'frozen','rollback_version':'base'}
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
            decision_context=read_json(next((child/'research_archive/decisions').glob('*/context.json')))
            self.assertEqual(decision_context['diagnostic']['status'],'success')
            self.assertEqual(set(decision_context['diagnostic']['output']['measurements']),{'unrelated'})
            self.assertEqual(decision_context['measurement_gaps'][0]['required_information'],'spread')
            self.assertFalse(any('hypothesis' in p.name for p in (child/'tasks').iterdir()))
            self.assertEqual(autoresearch(self.campaign,agent=agent),report)
        self.assertEqual(report['state']['measurement_obligations'][0]['required_information'],'spread')
        project=ResearchProject(self.campaign,agent=agent)
        try:self.assertEqual(len(project.verify()['verified_episodes']),3)
        finally:project.store.close()
