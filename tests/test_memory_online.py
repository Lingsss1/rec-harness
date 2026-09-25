"""Exercise complete paired controllers with injected decisions, never online claims."""
import unittest
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.memory_online import run_validation,fork_baseline
from rec_harness.online_validation import freeze_suite
from rec_harness.runtime import Research
import test_online_validation
from test_online_validation import PairedAgent


class MemoryOnlineTests(unittest.TestCase):
    setUp=test_online_validation.OnlineValidationTests.setUp
    evaluate=test_online_validation.OnlineValidationTests.evaluate
    def test_common_start_real_controllers_cost_resume_and_divergence(self):
        freeze_suite(self.root/'spec.json',self.root/'suite')
        args=(self.root/'suite/suite.json',self.root/'report.json','development',131072)
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.evaluate):
            report=run_validation(*args,agent_factory=lambda *_:PairedAgent())
            self.assertFalse(report['online']);self.assertEqual(report['total_cost'],15)
            row=report['rows'][0];self.assertEqual(row['shared_baseline_cost'],1)
            self.assertEqual(row['graph']['incremental_cost'],7)
            first=next(p for p in row['input_audit']['pairs'] if p['role']=='evidence')
            self.assertTrue(first['equal_visible_information'])
            self.assertFalse(row['input_audit']['all_inputs_equal'])
            for outcome in row['input_audit']['outcomes'].values():
                self.assertTrue(outcome['all_candidate_attempts_valid'])
                self.assertEqual(outcome['incomplete_roles'],[])
            def no_calls(*args):raise AssertionError('resume called agent factory')
            # Factory itself may be created on resume; its run must not execute.
            class NoCalls:
                run=no_calls
            self.assertEqual(run_validation(*args,agent_factory=lambda *_:NoCalls()),report)
            with self.assertRaisesRegex(ContractError,'inputs changed'):
                run_validation(*args)
        root=self.root/'memory-online/development'
        with self.assertRaisesRegex(ContractError,'pre-decision|trajectory'):
            fork_baseline(root/'graph',self.root/'illegal','flat')
        config=read_json(root/'flat/campaign.json');config['budget']+=1
        write_json(root/'flat/campaign.json',config)
        with self.assertRaises(ContractError):fork_baseline(root/'baseline',root/'flat','flat')

    def test_fork_refuses_unsettled_baseline_without_destination(self):
        freeze_suite(self.root/'spec.json',self.root/'suite')
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.evaluate):
            run_validation(self.root/'suite/suite.json',self.root/'report.json','development',131072,agent_factory=lambda *_:PairedAgent())
        baseline=self.root/'memory-online/development/baseline'
        research=Research(baseline,agent=object())
        try:research.store.reserve('unfinished',1)
        finally:research.store.close()
        with self.assertRaisesRegex(ContractError,'unsettled'):fork_baseline(baseline,self.root/'unsafe','flat')
        self.assertFalse((self.root/'unsafe').exists())

    def test_failed_candidates_are_exposed_even_when_terminal_scores_match(self):
        freeze_suite(self.root/'spec.json',self.root/'suite')
        def invalid(model,output,seed,*args):
            if 'attempt-1' in str(output):raise ContractError('fixture launcher failure')
            return self.evaluate(model,output,seed,*args)
        with patch('rec_harness.toy_isolated.evaluate',side_effect=invalid):
            report=run_validation(self.root/'suite/suite.json',self.root/'report.json','development',131072,agent_factory=lambda *_:PairedAgent())
        row=report['rows'][0];self.assertEqual(row['delta_graph_minus_flat'],0)
        for outcome in row['input_audit']['outcomes'].values():
            self.assertFalse(outcome['all_candidate_attempts_valid'])
            self.assertEqual(outcome['attempts'][0]['selection'],'invalid')
            self.assertEqual(outcome['attempts'][0]['run_statuses'][0]['error'],'fixture launcher failure')
