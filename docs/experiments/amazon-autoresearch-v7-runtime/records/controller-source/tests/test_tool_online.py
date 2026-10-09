"""Real controller orchestration with mocked execution, never online evidence."""
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.tool_online import freeze_suite,run_validation
from rec_harness.tool_exposure import audit_exposure
import test_online_validation as paired
from test_online_validation import PairedAgent


class ToolOnlineTests(unittest.TestCase):
    evaluate=paired.OnlineValidationTests.evaluate

    def setUp(self):
        paired.OnlineValidationTests.setUp(self)
        for group in ['existing','candidate']:
            directory=self.root/group;directory.mkdir(exist_ok=True)
            (directory/'diagnostic.py').write_text('def measure(request): return {}\n# '+group)
        spec=read_json(self.root/'spec.json')
        spec.update(existing_tool='existing',tool_files=['diagnostic.py'],
            tool_runtime={'cost':1,'timeout':10,'execution':{'recipe':{},'python':'unused','uid_registry':'unused'}})
        write_json(self.root/'spec.json',spec)
        freeze_suite(self.root/'spec.json',self.root/'suite')

    def execute(self,directory,*args,**kwargs):
        output=Path(directory)/'results/measurement.json'
        write_json(output,{'measurements':{'spread':0.25},'limitations':['aggregate only']})
        return {'process':{'status':'success'},'collection':{'artifacts':{'measurement.json':kwargs['artifacts'].put(output)}}}

    def run_groups(self):
        return run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json','utility',agent_factory=lambda *_:PairedAgent())

    def test_three_independent_controllers_cost_and_exact_resume(self):
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.evaluate),patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.execute) as worker:
            report=self.run_groups();self.assertTrue(report['passed']);self.assertEqual(report['evaluation_mode'],'end_to_end_candidate_generation');self.assertFalse(report['online'])
            self.assertEqual(report['gain'],0);self.assertEqual(report['total_cost'],26)
            groups=report['comparison_groups'];self.assertEqual(len({report['rows'][0][g]['graph_directory'] for g in groups}),3)
            self.assertEqual(worker.call_count,2);self.assertEqual(self.run_groups(),report);self.assertEqual(worker.call_count,2)
            for group in groups:
                campaign=Path(report['rows'][0][group]['graph_directory']).parent
                exposure=audit_exposure(campaign)
                self.assertEqual(len(exposure['exposures']),int(group!='no_tool'))
                if group!='no_tool':self.assertTrue(exposure['exposures'][0]['transport_completed'])
        (self.root/'candidate/diagnostic.py').write_text('changed')
        with self.assertRaisesRegex(ContractError,'inputs changed'):self.run_groups()

    def test_failed_measurement_cannot_pass_utility(self):
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.evaluate),patch('rec_harness.diagnostics.run_cpu_job',side_effect=ContractError('failed')):
            report=self.run_groups();self.assertFalse(report['passed']);self.assertEqual(report['total_cost'],26)

    def test_gain_uses_both_references_and_worst_regression(self):
        def fake_episode(case,policy,root,inputs,**kwargs):
            group=Path(root).name
            return {'score':{'no_tool':0.2,'existing_tool':0.9,'candidate_tool':0.8}[group],
                'protocol_hash':'same','charged_cost':3,'diagnostic_calls':int(group!='no_tool'),'diagnostic_failures':0}
        with patch('rec_harness.tool_online.episode',side_effect=fake_episode):
            report=self.run_groups()
        self.assertAlmostEqual(report['gain'],-0.1)
        self.assertAlmostEqual(report['regression'],0.1)
        self.assertEqual(report['total_cost'],9)
