"""Prefix recovery retains outcomes, costs and exact invocation identity."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,digest,write_json
from rec_harness.store import Store
from rec_harness.harness_prefix import run_validation
from rec_harness.validation_stage import report,ValidationRecoveryRequired
import test_harness_prefix


class PrefixRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_harness_prefix.PrefixTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.calls=[]
        outer=self
        class Agent:
            def run(self,role,context,directory):
                outer.calls.append(role);return outer.fixture.response
        self.factory=lambda *_:Agent()

    def run_prefix(self):
        return run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json',agent_factory=self.factory)

    def interrupt(self,store,kind,value):
        if kind=='prefix_replay_completed':raise KeyboardInterrupt()
        return self.original(store,kind,value)

    def test_result_committed_before_completion_is_reused_from_cas(self):
        self.original=Store.event
        with patch.object(Store,'event',lambda store,kind,value:self.interrupt(store,kind,value)):
            with self.assertRaises(KeyboardInterrupt):self.run_prefix()
        self.assertEqual(len(self.calls),1)
        write_json(self.root/'prefix-replay/cases/trace-one-baseline/attempt-1/replay-response.json',{'changed':True})
        result=self.run_prefix();self.assertTrue(result['passed']);self.assertEqual(result['total_cost'],2)
        self.assertEqual(len(self.calls),2);self.assertEqual(self.run_prefix(),result)
        self.assertEqual(len(self.calls),2)
        self.assertFalse(list((self.root/'prefix-replay/cases').glob('*/attempt-2')))

    def test_unknown_role_outcome_cannot_be_resubmitted(self):
        class Interrupted:
            def run(self,*args):raise KeyboardInterrupt()
        self.factory=lambda *_:Interrupted()
        with self.assertRaises(KeyboardInterrupt):self.run_prefix()
        for _ in range(2):
            with self.assertRaisesRegex(ContractError,'not committed'):self.run_prefix()
        store=Store(self.root/'prefix-replay/memory',read_only=True)
        try:self.assertEqual(store.budget()['charged_or_reserved'],1)
        finally:store.close()

    def test_outer_stage_reuses_prefix_result_or_blocks_unknown_before_new_charge(self):
        for unknown in [False,True]:
            with self.subTest(unknown=unknown):
                root=self.root/('unknown' if unknown else 'complete');store=Store(root/'memory')
                self.addCleanup(store.close);store.configure_budget(10);objects=Artifacts(root/'objects')
                implementation=objects.snapshot(self.root/'candidate',['policy.json'])
                candidate={'candidate_id':'candidate','plan_hash':'fixture','implementation':implementation,'implementation_hash':digest(implementation)}
                evolution=SimpleNamespace(root=root,store=store,artifacts=objects)
                plan={'budget':10,'stages':{'prefix_audit':{'cost':1,'timeout':30,
                    'command':['{python}','-m','rec_harness.harness_prefix','run']}}}
                def runner(argv,cwd,directory,**kwargs):
                    run_validation(self.root/'suite/suite.json',cwd/'candidate',cwd/'result.json',agent_factory=self.factory)
                    write_json(directory/'process.json',{'status':'success'});return {'status':'success'}
                original=Store.event
                def interrupt(store,kind,value):
                    if kind==('prefix_replay_submitted' if unknown else 'prefix_replay_completed'):
                        if unknown:original(store,kind,value)
                        raise KeyboardInterrupt()
                    return original(store,kind,value)
                with patch.object(Store,'event',interrupt):
                    with self.assertRaises(KeyboardInterrupt):report(evolution,candidate,plan,'prefix_audit',runner)
                if unknown:
                    for _ in range(2):
                        with self.assertRaises(ValidationRecoveryRequired):
                            report(evolution,candidate,plan,'prefix_audit',lambda *a,**kw:self.fail('relaunch'))
                    self.assertEqual(store.budget()['charged_or_reserved'],1)
                else:
                    result=report(evolution,candidate,plan,'prefix_audit',runner)
                    self.assertTrue(result['passed']);self.assertEqual(result['total_cost'],2)
                    self.assertEqual(len(self.calls),2);self.assertEqual(store.budget()['charged_or_reserved'],2)
