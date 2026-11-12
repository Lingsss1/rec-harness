import unittest
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.runtime import Research
import test_runtime


class RoundArchiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture=test_runtime.RuntimeTests();fixture.setUp();self.addCleanup(fixture.tearDown)
        self.root=fixture.path/'campaign';Research.init(self.root,fixture.config())
        self.research=Research(self.root,agent=object());self.addCleanup(self.research.store.close)
        self.state=read_json(self.root/'state.json')
        self.state['baseline']={'attempt':'baseline-fixture','valid':True,'score':0.5,'runs':[]}
        write_json(self.root/'state.json',self.state)
        self.research.store.event('round_started',{'attempt':'attempt-1','state':self.state})
        self.directory=self.root/'research_archive/attempts/attempt-1'

    def archive(self,decision):
        self.research._record_attempt('attempt-1',self.state,self.state['workspace'],self.state['workspace'],
            {'attempt':'attempt-1','valid':False,'score':None,'runs':[]},decision,[])

    def test_committed_measurement_pause_survives_projection_tampering_and_resumes_idle(self):
        decision={'selection':'provisional','conclusion_status':'unknown','measurement_required':True,'rationale':'missing measurement'}
        self.archive(decision)
        write_json(self.directory/'decision.json',{'selection':'keep','measurement_required':False})
        write_json(self.directory/'observations.json',{'valid':True,'score':999})
        with patch.object(self.research,'_agent',side_effect=AssertionError('new role')),patch.object(self.research,'evaluate',side_effect=AssertionError('new evaluation')):
            result=self.research.run()
            self.assertEqual(result['state']['stop_reason'],'measurement_required')
            self.assertEqual(result['state']['incumbent'],'root')
            self.assertEqual(read_json(self.directory/'decision.json'),decision)
            before=self.research.store.verify()
            self.assertEqual(self.research.run(),result)
            self.assertEqual(self.research.store.verify(),before)

    def test_uncommitted_keep_files_do_not_promote_a_model(self):
        write_json(self.directory/'node.json',{'workspace_after':self.state['workspace']})
        write_json(self.directory/'observations.json',{'attempt':'attempt-1','valid':True,'score':999,'runs':[]})
        write_json(self.directory/'decision.json',{'selection':'keep'})
        result=self.research.run(0)
        self.assertEqual(result['state']['incumbent'],'root')
        decision=read_json(self.directory/'decision.json')
        self.assertEqual(decision['selection'],'crash');self.assertEqual(decision['conclusion_status'],'unknown')

    def test_corrupt_committed_archive_stops_before_round_completion(self):
        self.archive({'selection':'crash','conclusion_status':'unknown'})
        record=next(e['payload'] for e in self.research.store.events() if e['kind']=='attempt_archived')
        self.research.artifacts.verify(record['archive']['decision.json']).write_text('{}')
        with self.assertRaisesRegex(ContractError,'corrupt artifact'):self.research.run(0)
        self.assertFalse(any(e['kind']=='round_completed' for e in self.research.store.events()))

    def test_committed_archive_must_match_started_round(self):
        changed={**self.state,'incumbent':'other-model'}
        self.research._record_attempt('attempt-1',changed,self.state['workspace'],self.state['workspace'],
            {'attempt':'attempt-1','valid':False,'score':None,'runs':[]},{'selection':'crash'},[])
        with self.assertRaisesRegex(ContractError,'differs from interrupted'):self.research.run(0)

    def test_project_recovery_preserves_model_to_tool_handoff(self):
        import test_project
        from rec_harness.autonomous import autoresearch
        from rec_harness.store import Store
        fixture=test_project.ProjectTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        original=Store.event
        def interrupt(store,kind,value):
            if kind=='round_completed' and value['decision'].get('measurement_required'):
                raise KeyboardInterrupt()
            return original(store,kind,value)
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=fixture.cpu):
            with patch.object(Store,'event',interrupt):
                with self.assertRaises(KeyboardInterrupt):autoresearch(fixture.campaign,fixture.config('tool'),agent=fixture.agent)
            first=fixture.campaign/'episodes/model-1/campaign'
            store=Store(first/'memory',read_only=True)
            try:before=sum(e['kind']=='evaluation_completed' for e in store.events())
            finally:store.close()
            result=autoresearch(fixture.campaign,agent=fixture.agent)
        self.assertEqual([e['kind'] for e in result['episodes']],['model','tool','model'])
        self.assertEqual(read_json(first/'run-report.json')['state']['stop_reason'],'measurement_required')
        store=Store(first/'memory',read_only=True)
        try:self.assertEqual(sum(e['kind']=='evaluation_completed' for e in store.events()),before)
        finally:store.close()
