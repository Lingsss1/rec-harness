import copy
import unittest
from pathlib import Path
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.store import Store
from rec_harness.tool_history import collect
from rec_harness.evolution_auto import AutonomousEvolution
from rec_harness.evolution import STAGES
import test_tool_method


class ToolHistoryTests(unittest.TestCase):
    def setUp(self):
        fixture=test_tool_method.ToolMethodTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.root=fixture.fixture.root;self.source=self.root/'episodes/history'
        self.store=Store(self.source/'memory');self.addCleanup(self.store.close)
        self.objects=Artifacts(self.source/'objects');self.plan={'development_ids':['old-development']}
        write_json(self.source/'candidates/old/plan.json',self.plan)
        self.store.event('candidate_proposed',{'candidate_id':'old','kind':'tool','plan_hash':digest(self.plan)})
        self.store.event('candidate_implemented',{'candidate_id':'old','implementation_hash':'old-code'})
        report={'evaluation_ids':['old-validation'],'gain':0.125,'private_fixture_marker':'MUST_NOT_SHARE'}
        self.store.event('validation_rejected',{'candidate_id':'old','stage':'utility','rejection_kind':'validation_gate',
            'report':report,'report_hash':digest(report)})
        self.store.event('evolution_stopped',{'candidate_id':'old','status':'rejected'})

    def test_project_history_uses_committed_terminal_prefix_and_checks_report(self):
        from types import SimpleNamespace
        from rec_harness.tool_history import project_sources
        project=SimpleNamespace(root=self.root,store=Store(self.root/'project-memory'))
        self.addCleanup(project.store.close)
        report={'candidate_id':'old','status':'rejected'}
        write_json(self.source/'run-report.json',report)
        entry={'kind':'tool','campaign':str(self.source),'result_hash':digest(report)}
        # An uncommitted directory is not an input to the next author.
        self.assertEqual(project_sources(project),[])
        project.store.event('project_transition',{'episode':entry})
        sources=project_sources(project)
        before=self.store.verify()
        self.store.event('evolution_stopped',{'candidate_id':'old','status':'future'})
        self.assertEqual(project_sources(project),sources)
        history,_=collect(sources)
        self.assertEqual(history['candidates'][0]['status'],'rejected')
        self.assertEqual(sources[0]['cutoff_seq'],before['events'])
        write_json(self.source/'run-report.json',{'status':'forged'})
        with self.assertRaisesRegex(ContractError,'report changed'):project_sources(project)

    def test_project_history_rejects_uncommitted_terminal_report(self):
        from types import SimpleNamespace
        from rec_harness.tool_history import project_sources
        project=SimpleNamespace(root=self.root,store=Store(self.root/'project-memory'))
        self.addCleanup(project.store.close)
        report={'candidate_id':'old','status':'invented'}
        write_json(self.source/'run-report.json',report)
        project.store.event('project_transition',{'episode':{
            'kind':'tool','campaign':str(self.source),'result_hash':digest(report)}})
        with self.assertRaisesRegex(ContractError,'committed terminal'):project_sources(project)

    def test_freeze_excludes_scores_future_events_and_duplicate_sources(self):
        before=self.store.verify();cutoff=before['events']
        history,excluded=collect([{'campaign':str(self.source),'cutoff_seq':cutoff}])
        self.assertEqual(self.store.verify(),before)
        self.assertEqual(excluded,['old-development','old-validation'])
        row=history['candidates'][0];self.assertEqual(row['status'],'rejected');self.assertIsNone(row['method_card'])
        self.assertNotIn('MUST_NOT_SHARE',str(history));self.assertNotIn('0.125',str(history))
        self.store.event('evolution_stopped',{'candidate_id':'old','status':'future'})
        self.assertEqual(collect([{'campaign':str(self.source),'cutoff_seq':cutoff}]),(history,excluded))
        with self.assertRaisesRegex(ContractError,'duplicate'):
            collect([{'campaign':str(self.source),'cutoff_seq':cutoff},{'campaign':str(self.source)}])

    def test_new_author_receives_frozen_history_and_exclusions(self):
        config=read_json(self.fixture.fixture.config());config['kind']='tool';config['context']={}
        rule=next(iter(config['plan']['stages'].values()))
        config['plan']['stages']={stage:copy.deepcopy(rule) for stage in STAGES['tool']}
        config['tool_history_sources']=[{'campaign':str(self.source)}]
        target=self.root/'new';AutonomousEvolution.init(target,config)
        frozen=read_json(target/'evolution-auto.json')
        self.assertNotIn('tool_history_sources',frozen)
        self.assertIn('old-validation',frozen['plan']['development_ids'])
        outer=self
        class Author:
            def run(self,role,context,directory):
                outer.assertEqual(context['context']['tool_history']['candidates'][0]['status'],'rejected')
                model=Path(directory)/'model';model.mkdir(parents=True)
                (model/'diagnostic.py').write_text('def measure(request): return {}\n')
                from test_evolution_auto import tool_definition
                return {'changed_files':['diagnostic.py']+tool_definition(context,directory)}
        self.store.event('evolution_stopped',{'candidate_id':'old','status':'future'})
        auto=AutonomousEvolution(target,agent=Author());self.addCleanup(auto.store.close)
        c=auto.config;candidate=auto.evolution.propose(c['candidate_id'],'tool',c['context'],c['plan'],agent=auto.agent)
        self.assertEqual(candidate['status'],'implemented')

    def test_missing_validation_identity_or_changed_plan_rejected(self):
        self.store.event('validation_rejected',{'candidate_id':'old','stage':'shadow'})
        with self.assertRaisesRegex(ContractError,'evaluation IDs'):collect([{'campaign':str(self.source)}])
        write_json(self.source/'candidates/old/plan.json',{'development_ids':['changed']})
        with self.assertRaisesRegex(ContractError,'plan changed'):collect([{'campaign':str(self.source)}])

    def test_method_card_source_is_verified_and_not_promoted_to_fact(self):
        plan={**self.plan,'tool_method_contract':'diagnostic-method-v1'}
        write_json(self.source/'candidates/new/plan.json',plan)
        write_json(self.root/'card.json',self.fixture.card);ref=self.objects.put(self.root/'card.json')
        self.store.event('candidate_proposed',{'candidate_id':'new','kind':'tool','plan_hash':digest(plan)})
        self.store.event('candidate_implemented',{'candidate_id':'new','implementation_hash':'new-code','proposal_ref':ref})
        history,_=collect([{'campaign':str(self.source),'candidate_ids':['new']}])
        self.assertEqual(history['candidates'][0]['method_card'],self.fixture.card)
        self.assertEqual(history['candidates'][0]['status'],'implemented')
        self.objects.verify(ref).write_text('{}')
        with self.assertRaisesRegex(ContractError,'corrupt artifact'):collect([{'campaign':str(self.source),'candidate_ids':['new']}])

    def test_superseded_and_rolled_back_are_not_reported_as_current_promotions(self):
        self.store.event('candidate_promoted',{'candidate_id':'old'})
        self.store.event('candidate_promoted',{'candidate_id':'replacement','previous':{'candidate_id':'old'}})
        self.assertEqual(collect([{'campaign':str(self.source)}])[0]['candidates'][0]['status'],'superseded')
        self.store.event('candidate_rollback',{'from':'old','to':None})
        self.assertEqual(collect([{'campaign':str(self.source)}])[0]['candidates'][0]['status'],'rolled_back')

    def test_project_preflights_history_before_creating_or_running_model(self):
        import test_project
        from rec_harness.project import ResearchProject
        fixture=test_project.ProjectTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=read_json(fixture.config('tool'));config['tool']['tool_history_sources']=[]
        with self.assertRaisesRegex(ContractError,'nonempty tool history'):
            ResearchProject.init(fixture.campaign,config)
        self.assertFalse(fixture.campaign.exists())

    def test_project_freezes_history_cutoff_before_first_model(self):
        import test_project
        from rec_harness.project import ResearchProject
        fixture=test_project.ProjectTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=read_json(fixture.config('tool'))
        config['tool']['tool_history_sources']=[{'campaign':str(self.source)}]
        cutoff=self.store.verify()['events']
        ResearchProject.init(fixture.campaign,config)
        frozen=read_json(fixture.campaign/'project.json')['tool']
        self.assertEqual(frozen['tool_history_sources'][0]['cutoff_seq'],cutoff)
        self.assertEqual(frozen['tool_history_sources'][0]['event_head'],self.store.verify()['head'])
        self.assertIn('old-validation',frozen['plan']['development_ids'])
        self.store.event('evolution_stopped',{'candidate_id':'old','status':'future'})
        history,_=collect(frozen['tool_history_sources'])
        self.assertEqual(history['candidates'][0]['status'],'rejected')
        wrong={**frozen['tool_history_sources'][0],'event_head':'0'*64}
        with self.assertRaisesRegex(ContractError,'event head changed'):collect([wrong])
