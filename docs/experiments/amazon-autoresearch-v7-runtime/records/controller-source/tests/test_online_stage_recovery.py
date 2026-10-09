"""Online stage restarts only at durable, fully completed group boundaries."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import Artifacts,digest,read_json,write_json
from rec_harness.store import Store
from rec_harness.validation_stage import report,ValidationRecoveryRequired
import test_online_validation


class OnlineStageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_online_validation.OnlineValidationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        from rec_harness.online_validation import freeze_suite
        freeze_suite(self.root/'spec.json',self.root/'suite')
        store=Store(self.root/'outer/memory');self.addCleanup(store.close);store.configure_budget(10)
        artifacts=Artifacts(self.root/'outer/objects')
        implementation=artifacts.snapshot(self.root/'candidate',['policy.json'])
        self.candidate={'candidate_id':'candidate','plan_hash':'fixture','implementation':implementation,
                        'implementation_hash':digest(implementation)}
        self.evolution=SimpleNamespace(root=self.root/'outer',store=store,artifacts=artifacts)
        self.plan={'budget':10,'stages':{'development':{'cost':1,'timeout':30,
            'command':['{python}','-m','rec_harness.online_validation','run','--candidate','{candidate}','--output','{output}']}}}
        self.calls=[]

    def runner(self,argv,cwd,directory,**kwargs):
        from rec_harness.online_validation import run_validation
        outer=self
        class Agent(test_online_validation.PairedAgent):
            def run(self,role,context,directory):
                outer.calls.append(role)
                return super().run(role,context,directory)
        run_validation(self.root/'suite/suite.json',cwd/'candidate',cwd/'result.json','development',agent_factory=lambda *_:Agent())
        write_json(directory/'process.json',{'status':'success'})
        return {'status':'success'}

    def run_stage(self,runner=None):
        return report(self.evolution,self.candidate,self.plan,'development',runner or self.runner)

    def test_completed_group_reused_before_new_group_without_duplicate_role_calls(self):
        from rec_harness.online_validation import episode
        def interrupt(*args,**kwargs):
            episode(*args,**kwargs)
            raise KeyboardInterrupt()
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.fixture.evaluate):
            with patch('rec_harness.online_validation.episode',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):self.run_stage()
            self.assertEqual(len(self.calls),6)
            result=self.run_stage()
            self.assertEqual(len(self.calls),12);self.assertEqual(result['total_cost'],16)
            self.assertFalse(result['online']);self.assertEqual(result['gain'],0)
            self.assertEqual(self.evolution.store.budget()['charged_or_reserved'],2)
            self.assertEqual(self.run_stage(lambda *a,**kw:self.fail('relaunch')),result)

    def test_incomplete_group_blocks_before_wrapper_launch_and_new_charge(self):
        original=Store.event
        def interrupt(store,kind,value):
            if kind=='agent_submitted':raise KeyboardInterrupt()
            return original(store,kind,value)
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.fixture.evaluate),patch.object(Store,'event',interrupt):
            with self.assertRaises(KeyboardInterrupt):self.run_stage()
        before=self.evolution.store.budget()
        for _ in range(2):
            with self.assertRaisesRegex(ValidationRecoveryRequired,'not durably completed'):
                self.run_stage(lambda *a,**kw:self.fail('relaunch'))
        self.assertEqual(self.evolution.store.budget(),before);self.assertEqual(self.calls,[])

    def test_changed_terminal_projection_cannot_authorize_online_restart(self):
        from rec_harness.online_validation import episode
        def interrupt(*args,**kwargs):
            episode(*args,**kwargs);raise KeyboardInterrupt()
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.fixture.evaluate),patch('rec_harness.online_validation.episode',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):self.run_stage()
        path=next((self.root/'outer').glob('candidates/*/validation/development/online-episodes/*/baseline/state.json'))
        state=read_json(path);state['baseline']['score']=999;write_json(path,state)
        with self.assertRaisesRegex(ValidationRecoveryRequired,'not durably completed'):
            self.run_stage(lambda *a,**kw:self.fail('relaunch'))
        self.assertEqual(len(self.calls),6)

    def test_tool_utility_resumes_after_two_groups_without_repeating_diagnostics(self):
        import test_tool_online
        from rec_harness.tool_online import run_validation
        from rec_harness.online_validation import episode
        for module in ['rec_harness.tool_online','rec_harness.tool_pipeline']:
            with self.subTest(module=module):
                fixture=test_tool_online.ToolOnlineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
                root=fixture.root/'outer';store=Store(root/'memory');self.addCleanup(store.close)
                store.configure_budget(10);artifacts=Artifacts(root/'objects')
                implementation=artifacts.snapshot(fixture.root/'candidate',['diagnostic.py'])
                candidate={**self.candidate,'implementation':implementation,'implementation_hash':digest(implementation)}
                evolution=SimpleNamespace(root=root,store=store,artifacts=artifacts)
                plan={'budget':10,'stages':{'utility':{'cost':1,'timeout':30,
                    'command':['{python}','-m',module,'run','--candidate','{candidate}','--output','{output}']}}}
                def runner(argv,cwd,directory,**kwargs):
                    run_validation(fixture.root/'suite/suite.json',cwd/'candidate',cwd/'result.json','utility',
                                   agent_factory=lambda *_:test_online_validation.PairedAgent())
                    write_json(directory/'process.json',{'status':'success'});return {'status':'success'}
                def interrupt(*args,**kwargs):
                    result=episode(*args,**kwargs)
                    if args[2].name=='existing_tool':raise KeyboardInterrupt()
                    return result
                with patch('rec_harness.toy_isolated.evaluate',side_effect=fixture.evaluate),patch('rec_harness.diagnostics.run_cpu_job',side_effect=fixture.execute) as worker:
                    with patch('rec_harness.tool_online.episode',side_effect=interrupt):
                        with self.assertRaises(KeyboardInterrupt):report(evolution,candidate,plan,'utility',runner)
                    self.assertEqual(worker.call_count,1)
                    result=report(evolution,candidate,plan,'utility',runner)
                    self.assertTrue(result['passed']);self.assertFalse(result['online'])
                    self.assertEqual(result['total_cost'],26);self.assertEqual(worker.call_count,2)
                    self.assertEqual(store.budget()['charged_or_reserved'],2)

    def test_harness_pipeline_repeated_resume_freezes_each_invocation_cost(self):
        import test_harness_pipeline
        from rec_harness.harness_pipeline import run_validation as pipeline
        from rec_harness.online_validation import run_validation as online,episode
        from rec_harness.cost_audit import verify_cost_prefix
        fixture=test_harness_pipeline.HarnessPipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        rule=self.plan['stages']['development']
        rule['command']=['{python}','-m','rec_harness.harness_pipeline','run','--cost-context','{cost_context}']
        target=self.evolution.root/'candidates/candidate/validation/development'
        previous={'identity':fixture.online['identity'],'pipeline_hash':read_json(fixture.suite)['hash']}
        write_json(target.parent/'prefix_audit/accepted.json',{'report':previous,'report_hash':digest(previous),
            'implementation_hash':self.candidate['implementation_hash']})
        calls=[]
        class Agent(test_online_validation.PairedAgent):
            def run(self,role,context,directory):
                calls.append(role);return super().run(role,context,directory)
        def online_with_agent(*args,**kwargs):
            return online(*args,**kwargs,agent_factory=lambda *_:Agent())
        def runner(argv,cwd,directory,**kwargs):
            pipeline(fixture.suite,'development',cwd/'candidate',cwd/'result.json',argv[-1])
            write_json(directory/'process.json',{'status':'success'});return {'status':'success'}
        def interrupt(*args,**kwargs):
            episode(*args,**kwargs);raise KeyboardInterrupt()
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.fixture.evaluate),patch('rec_harness.online_validation.run_validation',side_effect=online_with_agent):
            for _ in range(2):
                with patch('rec_harness.online_validation.episode',side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):self.run_stage(runner)
                self.assertEqual(len(calls),6)
            result=self.run_stage(runner)
        self.assertEqual(len(calls),12);self.assertFalse(result['online'])
        self.assertEqual(result['stage_cost'],16);self.assertEqual(result['total_cost'],19)
        self.assertEqual(self.evolution.store.budget()['charged_or_reserved'],3)
        for index,name in enumerate(['controller-cost.json','controller-cost-resume-2.json','controller-cost-resume-3.json'],1):
            prefix=read_json(target/name)
            self.assertEqual(verify_cost_prefix(self.evolution.store,prefix),index)
            self.assertEqual(prefix['pending'],1)
        self.assertEqual(self.run_stage(lambda *a,**kw:self.fail('relaunch')),result)
