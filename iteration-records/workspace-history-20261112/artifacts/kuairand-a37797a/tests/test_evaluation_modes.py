import copy
import unittest
from rec_harness.core import ContractError,read_json
from rec_harness.evolution import Evolution,ValidationGateRejected
import test_evolution_auto


class EvaluationModeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_evolution_auto.EvolutionAutomationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan=read_json(self.fixture.config())['plan']
        self.evolution=Evolution(self.fixture.campaign);self.addCleanup(self.evolution.store.close)
        self.candidate={'kind':'harness','candidate_id':'fixture'}
        self.report={'passed':True,'identity':self.plan['identity'],'gain':1,'regression':0,'total_cost':4,
            'evaluation_ids':['independent'],'comparison_groups':['baseline','candidate'],'online':True}

    def check(self,report):
        return self.evolution._check_report(self.candidate,self.plan,'development',report)

    def test_selection_score_cannot_satisfy_generation_contract(self):
        self.plan['stages']['development']['evaluation_mode']='end_to_end_candidate_generation'
        for value in [None,'fixed_candidate_selection','offline_measurement']:
            with self.assertRaisesRegex(ContractError,'evaluation mode differs') as error:
                self.check({**self.report,'evaluation_mode':value})
            self.assertNotIsInstance(error.exception,ValidationGateRejected)
        self.check({**self.report,'evaluation_mode':'end_to_end_candidate_generation'})
        self.plan['stages']['development']['evaluation_mode']='fixed_candidate_selection'
        self.check({**self.report,'evaluation_mode':'fixed_candidate_selection'})

    def test_legacy_untyped_plan_does_not_gain_new_requirements(self):
        self.check(self.report)

    def test_invalid_mode_rejected_before_author_call(self):
        for value in ['unknown',None,[]]:
            plan=copy.deepcopy(self.plan);plan['stages']['development']['evaluation_mode']=value
            from rec_harness.validation_preflight import preflight
            with self.assertRaisesRegex(ContractError,'unknown validation evaluation mode'):preflight(plan)
            with self.assertRaisesRegex(ContractError,'unknown validation evaluation mode'):
                self.evolution.propose('candidate','harness',{},plan,agent=self.fixture.agent)
        self.assertEqual(self.fixture.agent.calls,0)
