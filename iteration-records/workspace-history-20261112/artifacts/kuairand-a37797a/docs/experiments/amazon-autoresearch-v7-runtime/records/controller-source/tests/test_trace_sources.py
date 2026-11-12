import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.autonomous import autoresearch,configuration
from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.decision_traces import collect_traces,collect_trace_set,source_task_ids
from rec_harness.evolution_auto import AutonomousEvolution
from rec_harness.optimizer_handoff import request
from rec_harness.trace_context import compact_traces,expand_traces
from rec_harness.store import Store
from test_runtime import ROOT,ScriptedAgent


class TraceSourcesTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.sources=[]
        for name in ['left','right']:
            config=read_json(ROOT/'examples/toy/campaign.json')
            config.update(task_id=name,source=str(ROOT/'examples/toy'),max_rounds=1)
            path=self.root/(name+'.json');write_json(path,config)
            campaign=self.root/name;autoresearch(campaign,path,agent=ScriptedAgent())
            packet=collect_traces(campaign)
            self.sources.append({'campaign':str(campaign),'cutoff_seq':packet['source']['cutoff_seq']})

    def test_colliding_local_ids_preserve_exact_sources_and_prefixes(self):
        packets=[collect_traces(row['campaign']) for row in self.sources]
        self.assertEqual([t['trace_id'] for t in packets[0]['decision_traces']],[t['trace_id'] for t in packets[1]['decision_traces']])
        collection=collect_trace_set(self.sources)
        self.assertEqual(len({t['trace_id'] for t in collection['decision_traces']}),12)
        self.assertEqual(source_task_ids(collection),{'left','right'})
        self.assertEqual(collect_trace_set(list(reversed(self.sources))),collection)
        originals={p['hash']:{t['trace_id']:t for t in p['decision_traces']} for p in packets}
        for trace in collection['decision_traces']:
            restored=copy.deepcopy(trace);restored['trace_id']=restored.pop('source_trace_id')
            prefix=restored.pop('source_prefix_hash');expected_hash=restored.pop('source_trace_hash')
            self.assertEqual(restored,originals[prefix][restored['trace_id']])
            self.assertEqual(digest(restored),expected_hash)
        self.assertEqual(expand_traces(compact_traces(collection)),collection)
        store=Store(Path(self.sources[0]['campaign'])/'memory')
        try:store.event('later_annotation',{'note':'outside frozen prefix'})
        finally:store.close()
        self.assertEqual(collect_trace_set(self.sources),collection)

    def test_duplicate_paths_or_cloned_evidence_rejected(self):
        with self.assertRaisesRegex(ContractError,'duplicate trace source'):collect_trace_set([self.sources[0],self.sources[0]])
        clone=self.root/'clone';shutil.copytree(self.root/'left',clone)
        with self.assertRaisesRegex(ContractError,'duplicate role evidence'):collect_trace_set([self.sources[0],{'campaign':str(clone)}])

    def test_all_source_tasks_excluded_before_evolution_creation(self):
        config={'candidate_id':'multi','kind':'harness','trace_sources':self.sources,
                'plan':{'development_ids':['left']},'optimizer_handoff_budget':5}
        target=self.root/'evolution'
        with self.assertRaisesRegex(ContractError,'every trace source'):AutonomousEvolution.init(target,config)
        self.assertFalse(target.exists())
        config['plan']['development_ids'].append('right');AutonomousEvolution.init(target,config)
        frozen=expand_traces(read_json(target/'evolution-auto.json')['context'])
        self.assertEqual(source_task_ids(frozen),{'left','right'})
        packet=frozen['measurement_handoff_request'];self.assertIsNone(packet['diagnostic'])
        self.assertEqual({row['task_id'] for row in packet['trace_evidence']},{'left','right'})
        self.assertTrue(all('protocol_hash' in row and 'source_prefix_hash' in row for row in packet['trace_evidence']))

    def test_relative_paths_and_ambiguous_configuration(self):
        config={'adapter':'research-evolution-v1','trace_sources':[{'campaign':'left'},{'campaign':'right','roles':['evidence']}],
                'kind':'harness','candidate_id':'multi','plan':{'development_ids':['left','right']}}
        path=self.root/'config.json';write_json(path,config);resolved=configuration(path)
        self.assertEqual(resolved['trace_sources'][0]['campaign'],str(self.root/'left'))
        self.assertEqual(resolved['trace_sources'][1]['roles'],['evidence'])
        resolved['trace_source']=self.sources[0]
        with self.assertRaisesRegex(ContractError,'not both'):AutonomousEvolution.init(self.root/'invalid',resolved)

    def test_measurement_handoff_retains_all_source_scopes_and_exclusions(self):
        import test_project
        from test_optimizer_handoff import ReferringAgent
        from rec_harness.optimizer_handoff import load_context
        config=read_json(test_project.ProjectTests.config(self,'harness'))['harness']
        config.update(candidate_id='multi-handoff',trace_sources=self.sources,optimizer_handoff_budget=10)
        config['plan']['development_ids']=['left','right']
        path=self.root/'multi.json';write_json(path,config);target=self.root/'handoff'
        report=autoresearch(target,path,agent=ReferringAgent())
        self.assertEqual(report['status'],'measurement_required')
        context=load_context(target)
        self.assertEqual(source_task_ids(context),{'left','right'})
        self.assertEqual({row['task_id'] for row in context['observations']},{'left','right'})
        self.assertEqual(load_context(target),context)
        tool=read_json(test_project.ProjectTests.config(self,'tool'))['tool']
        tool.update(candidate_id='multi-tool',optimizer_measurement_source=str(target))
        tool['plan']['development_ids']=['left']
        with self.assertRaisesRegex(ContractError,'every optimizer source'):AutonomousEvolution.init(self.root/'tool-evolution',tool)
        self.assertFalse((self.root/'tool-evolution').exists())
        tool['plan']['development_ids'].append('right')
        AutonomousEvolution.init(self.root/'tool-evolution',tool)
        self.assertEqual(source_task_ids(read_json(self.root/'tool-evolution/evolution-auto.json')['context']),{'left','right'})

    def test_project_combines_declared_sources_with_current_episode(self):
        import test_project
        path=test_project.ProjectTests.config(self,'harness');config=read_json(path)
        config['harness']['trace_sources']=self.sources;write_json(path,config)
        project=self.root/'combined-project'
        report=autoresearch(project,path,agent=test_project.ProjectAgent())
        self.assertEqual([row['kind'] for row in report['episodes']],['model','harness','model'])
        frozen=read_json(project/'episodes/harness-1/campaign/evolution-auto.json')
        self.assertEqual(source_task_ids(frozen['context']),{'left','right','project-model-1'})
        self.assertTrue(source_task_ids(frozen['context']).issubset(frozen['plan']['development_ids']))

    def test_project_rejects_invalid_history_before_any_role_or_directory(self):
        import test_project
        from rec_harness.project import ResearchProject
        original=read_json(test_project.ProjectTests.config(self,'harness'))
        for sources,ambiguous in [(self.sources,True),([],False),
                                  ([self.sources[0],self.sources[0]],False)]:
            with self.subTest(sources=sources,ambiguous=ambiguous):
                config=copy.deepcopy(original)
                config['harness']['trace_sources']=sources
                if ambiguous:config['harness']['trace_source']=self.sources[0]
                target=self.root/'invalid-project'
                with self.assertRaises(ContractError):ResearchProject.init(target,config)
                self.assertFalse(target.exists())
        config=copy.deepcopy(original)
        config['harness']['trace_sources']=[{'campaign':str(self.root/'missing-history')}]
        config['max_harness_updates']=0
        ResearchProject.init(self.root/'disabled-project',config)

    def test_project_referral_carries_all_source_exclusions_to_tool(self):
        import test_project
        from test_optimizer_handoff import ReferringAgent
        config=read_json(test_project.ProjectTests.config(self,'harness'))
        extra=read_json(test_project.ProjectTests.config(self,'tool'))
        config.update(tool=extra['tool'],tool_runtime=extra['tool_runtime'])
        config['model']['measurement_discovery']=None
        config['harness']['trace_sources']=self.sources
        path=self.root/'combined-referral.json';write_json(path,config)
        project=self.root/'referral-project'
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=lambda *a,**k:test_project.ProjectTests.cpu(self,*a,**k)):
            report=autoresearch(project,path,agent=ReferringAgent())
        self.assertEqual([row['kind'] for row in report['episodes']],['model','harness','tool','model'])
        frozen=read_json(project/'episodes/tool-1/campaign/evolution-auto.json')
        self.assertEqual(source_task_ids(frozen['context']),{'left','right','project-model-1'})
        self.assertTrue(source_task_ids(frozen['context']).issubset(frozen['plan']['development_ids']))
