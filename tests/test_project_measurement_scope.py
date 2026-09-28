import unittest
from unittest.mock import patch
from rec_harness.autonomous import autoresearch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.project import ResearchProject
import test_project
import test_optimizer_handoff


class ScopeTests(unittest.TestCase):
    setUp=test_project.ProjectTests.setUp
    cpu=test_project.ProjectTests.cpu

    def configuration(self,output='other'):
        path=test_project.ProjectTests.config(self,'tool');config=read_json(path)
        config['model'].pop('measurement_requirements',None)
        config['tool']['plan']['measurement_scope']={'supported_outputs':{output:'Declared fixture quantity'}}
        write_json(path,config);return path

    def test_discovered_gap_retained_before_tool_call_and_terminal_resume(self):
        path=self.configuration();agent=test_project.DiscoverAgent();original=agent.run;calls=[]
        def invoke(role,*args):
            calls.append(role)
            if role=='tool':raise AssertionError('unsupported tool generation')
            return original(role,*args)
        agent.run=invoke
        result=autoresearch(self.campaign,path,agent=agent)
        self.assertEqual(len(agent.measurement_calls),1)
        self.assertEqual(result['state']['stop_reason'],'measurement_contract_unsupported')
        self.assertEqual(result['state']['pending_gap']['required_information'],'spread')
        self.assertEqual(result['state']['measurement_contract_mismatch']['supported_outputs'],['other'])
        self.assertEqual([e['kind'] for e in result['episodes']],['model'])
        before=list(calls)
        self.assertEqual(autoresearch(self.campaign,agent=agent),result)
        self.assertEqual(calls,before)
        self.assertFalse((self.campaign/'episodes/tool-1').exists())

    def test_matching_declared_quantity_uses_existing_validation_path(self):
        path=self.configuration('spread');agent=test_project.DiscoverAgent()
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.cpu):
            report=autoresearch(self.campaign,path,agent=agent)
        self.assertEqual([e['kind'] for e in report['episodes']],['model','tool','model'])
        self.assertNotIn('measurement_contract_mismatch',report['state'])

    def test_invalid_scope_rejected_before_project_creation(self):
        path=self.configuration();original=read_json(path)
        for scope in [True,{}, {'supported_outputs':[]},{'supported_outputs':{}},
                      {'supported_outputs':{'spread':''}},{'supported_outputs':{'': 'invalid'}}]:
            with self.subTest(scope=scope):
                config={**original,'tool':{**original['tool'],'plan':{**original['tool']['plan'],'measurement_scope':scope}}}
                with self.assertRaisesRegex(ContractError,'scope requires'):ResearchProject.init(self.campaign,config)
                self.assertFalse(self.campaign.exists())

    def test_optimizer_referral_stops_before_tool_and_retains_handoff(self):
        path=test_optimizer_handoff.OptimizerHandoffTests.configuration(self)
        config=read_json(path);config['tool']['plan']['measurement_scope']={'supported_outputs':{'other':'Fixture'}}
        write_json(path,config)
        agent=test_optimizer_handoff.ReferringAgent()
        report=autoresearch(self.campaign,path,agent=agent)
        self.assertEqual(report['state']['stop_reason'],'measurement_contract_unsupported')
        self.assertIn('pending_optimizer_handoff',report['state'])
        self.assertEqual([e['kind'] for e in report['episodes']],['model','harness'])
        self.assertNotIn('tool',agent.calls)

    def test_legacy_absent_flag_does_not_change_routing(self):
        project=object.__new__(ResearchProject)
        project.config={'tool':{'plan':{'measurement_scope':{'supported_outputs':{'other':'Fixture'}}}}}
        state={'status':'ready','phase':'model'}
        project.route_tool(state,{'required_information':'spread'})
        self.assertEqual(state['phase'],'tool')
        self.assertEqual(state['status'],'ready')
